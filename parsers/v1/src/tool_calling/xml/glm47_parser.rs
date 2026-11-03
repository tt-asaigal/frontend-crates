// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// GLM-4.7 Tool Call Parser
// Format: <tool_call>function_name<arg_key>param1</arg_key><arg_value>value1</arg_value></tool_call>
// Reference: https://huggingface.co/zai-org/GLM-4.7/blob/main/chat_template.jinja

use regex::Regex;
use serde_json::Value;
use std::collections::HashMap;
use tracing::warn;
use uuid::Uuid;

use super::super::ToolDefinition;
use super::super::config::Glm47ParserConfig;
use super::OrderedArguments;
use super::parsed_value::{ParsedValue, coerce_integer_literal};
use super::response::{CalledFunction, ToolCallResponse, ToolCallType};

/// Render a tool_call block snippet for logs. Bounded so a huge truncated
/// argument body doesn't blow up the log line; control chars are escaped
/// because raw newlines/tabs make the warning unreadable in grep/jq.
fn truncate_for_log(s: &str) -> String {
    const MAX: usize = 200;
    let mut out = String::with_capacity(MAX.min(s.len()) + 16);
    let mut bytes = 0usize;
    for ch in s.chars() {
        if bytes >= MAX {
            out.push('…');
            break;
        }
        match ch {
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c => out.push(c),
        }
        bytes += ch.len_utf8();
    }
    out
}

/// Check if a chunk contains the start of a GLM-4.7 tool call.
/// Format: <tool_call>function_name<arg_key>...</arg_key><arg_value>...</arg_value></tool_call>
pub fn detect_tool_call_start_glm47(chunk: &str, config: &Glm47ParserConfig) -> bool {
    let start_token = &config.tool_call_start;
    let arg_key_start = &config.arg_key_start;

    // Check if we have the complete start token
    if chunk.contains(start_token.as_str()) || chunk.contains(arg_key_start.as_str()) {
        return true;
    }

    // Check for partial match at the end of the chunk (for streaming)
    for i in 1..start_token.len() {
        if chunk.ends_with(&start_token[..i]) {
            return true;
        }
    }

    false
}

/// Find the end position of all consecutive GLM-4.7 tool calls.
/// When a model emits multiple parallel tool calls in one chunk
/// (e.g. `<tool_call>A</tool_call><tool_call>B</tool_call>`), this
/// function advances past every consecutive start→end pair so the
/// entire group is captured as a single jailed region.  Returns the
/// position after the last `</tool_call>` found, or the length of the
/// chunk when no end token is present.
pub fn find_tool_call_end_position_glm47(chunk: &str, config: &Glm47ParserConfig) -> usize {
    let start_token = &config.tool_call_start;
    let end_token = &config.tool_call_end;

    if !chunk.contains(start_token.as_str()) && chunk.contains(config.arg_key_start.as_str()) {
        return find_bare_glm47_tool_call_end_position(chunk, config).unwrap_or(chunk.len());
    }

    let Some(first_end) = chunk.find(end_token.as_str()) else {
        return chunk.len();
    };

    let mut cursor = first_end + end_token.len();

    loop {
        let rest = &chunk[cursor..];
        let trimmed = rest.trim_start();
        if !trimmed.starts_with(start_token.as_str()) {
            break;
        }
        let trim_offset = rest.len() - trimmed.len();
        let search_from = cursor + trim_offset + start_token.len();
        if let Some(end_pos) = chunk[search_from..].find(end_token.as_str()) {
            cursor = search_from + end_pos + end_token.len();
        } else {
            break;
        }
    }

    cursor
}

fn find_bare_glm47_tool_call_end_position(text: &str, config: &Glm47ParserConfig) -> Option<usize> {
    let marker_idx = first_orphan_glm47_marker_index(text, config)?;
    let before_marker = text[..marker_idx].trim_end();
    let function_name_start = before_marker
        .char_indices()
        .rev()
        .find(|(_, ch)| ch.is_whitespace())
        .map(|(idx, ch)| idx + ch.len_utf8())
        .unwrap_or(0);

    let candidate_name = before_marker[function_name_start..].trim();
    if candidate_name.is_empty()
        || !candidate_name
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '.'))
    {
        return None;
    }

    let mut cursor = function_name_start;
    let mut last_complete_end = None;
    while cursor < text.len() {
        let rest = &text[cursor..];
        let trim_offset = rest.len() - rest.trim_start().len();
        let call_start = cursor + trim_offset;
        let tail = &text[call_start..];
        let Some(end_pos) = tail.find(config.tool_call_end.as_str()) else {
            break;
        };

        let call_end = call_start + end_pos + config.tool_call_end.len();
        cursor = consume_glm47_close_markers(text, call_end, config);
        last_complete_end = Some(cursor);

        if first_orphan_glm47_marker_index(&text[cursor..], config).is_none() {
            break;
        }
    }

    last_complete_end
}

/// Try to parse GLM-4.7 formatted tool calls from a message.
/// Format: <tool_call>function_name<arg_key>param1</arg_key><arg_value>value1</arg_value></tool_call>
/// Returns (parsed_tool_calls, normal_text_content)
pub fn try_tool_call_parse_glm47(
    message: &str,
    config: &Glm47ParserConfig,
    tools: Option<&[ToolDefinition]>,
) -> anyhow::Result<(Vec<ToolCallResponse>, Option<String>)> {
    let (normal_text, tool_calls) = extract_tool_calls(message, config, tools)?;

    let normal_content = if normal_text.is_empty() {
        Some("".to_string())
    } else {
        Some(normal_text)
    };

    Ok((tool_calls, normal_content))
}

/// Extract tool calls and normal text from message.
fn extract_tool_calls(
    text: &str,
    config: &Glm47ParserConfig,
    tools: Option<&[ToolDefinition]>,
) -> anyhow::Result<(String, Vec<ToolCallResponse>)> {
    let mut normal_parts = Vec::new();
    let mut calls = Vec::new();
    let mut cursor = 0;

    let start_token = &config.tool_call_start;
    let end_token = &config.tool_call_end;

    if !text.contains(start_token.as_str())
        && let Some(marker_idx) = first_orphan_glm47_marker_index(text, config)
    {
        if let Some((prefix, mut parsed_calls)) =
            recover_bare_glm47_calls(text, marker_idx, config, tools)?
        {
            let recovered_calls = parsed_calls.len();
            warn!(
                why = "bare_body_recovery",
                recovered_calls,
                recovered_bytes = text.len() - prefix.len(),
                kept_prefix_bytes = prefix.len(),
                "GLM-4.7 parser recovered complete bare call body/bodies without <tool_call> start"
            );
            calls.append(&mut parsed_calls);
            return Ok((prefix, calls));
        }
        warn!(
            why = "GLM-4.7 tool-call marker found without <tool_call> start; dropping orphan marker tail so wire tags do not leak into normal_text",
            dropped_block = %truncate_for_log(&text[marker_idx..]),
            "GLM-4.7 parser dropping orphan tool-call marker tail"
        );
        return Ok((orphan_glm47_prefix(text, marker_idx), calls));
    }

    while cursor < text.len() {
        // Find next tool call start
        if let Some(start_pos) = text[cursor..].find(start_token.as_str()) {
            let abs_start = cursor + start_pos;
            let gap = &text[cursor..abs_start];
            if let Some((prefix, mut parsed_calls)) =
                recover_bare_glm47_calls_in_span(gap, config, tools)?
            {
                if calls.is_empty() {
                    normal_parts.push(prefix);
                }
                calls.append(&mut parsed_calls);
            } else if calls.is_empty() {
                if let Some(marker_idx) = first_orphan_glm47_marker_index(gap, config) {
                    normal_parts.push(orphan_glm47_prefix(gap, marker_idx));
                } else {
                    normal_parts.push(gap.to_string());
                }
            }

            // Only surface normal text that precedes the first parsed call.
            // Text after any </tool_call> is not response content; matches the
            // convention ported into the generic XML parser by PR #9350 and
            // vLLM's glm47_moe_tool_parser.

            // Find the corresponding end token
            if let Some(end_pos) = text[abs_start..].find(end_token.as_str()) {
                let abs_end = abs_start + end_pos + end_token.len();
                let block = &text[abs_start..abs_end];

                // Parse this tool call block. Unparseable blocks (malformed
                // <tool_call>...</tool_call> markup the parser can't extract)
                // are dropped — emitting the raw markup as normal_text leaks
                // wire tags downstream. vLLM and SGLang both drop on this
                // path; aligning Dynamo to that contract.
                match parse_tool_call_block(block, config, tools) {
                    Ok(parsed_call) => calls.push(parsed_call),
                    Err(e) => {
                        warn!(
                            reason = %e,
                            why = "block has open + close fence but content failed to parse \
                                   as a GLM-4.7 tool call (e.g. empty function name, \
                                   missing <arg_key>, malformed args); dropping to avoid \
                                   leaking wire tags through normal_text",
                            dropped_block = %truncate_for_log(block),
                            "GLM-4.7 parser dropping unparseable tool_call block"
                        );
                    }
                }

                cursor = abs_end;
            } else {
                // Recovery: outer </tool_call> absent (max_tokens / EOS
                // truncation). Gated on `allow_eof_recovery` so streaming
                // early-exit doesn't fire mid-stream. Also requires an
                // `<arg_key>` opener in the trailing slice as the structural
                // signal that a real tool call was emitted.
                let block = &text[abs_start..];
                let arg_key_start = &config.arg_key_start;
                if config.allow_eof_recovery && block.contains(arg_key_start.as_str()) {
                    match parse_tool_call_block(block, config, tools) {
                        Ok(parsed_call) => {
                            calls.push(parsed_call);
                            cursor = text.len();
                            continue;
                        }
                        Err(e) => {
                            warn!(
                                reason = %e,
                                why = "EOF recovery enabled and <arg_key> opener present, \
                                       but parse_tool_call_block failed on the truncated \
                                       tail; dropping to avoid leaking wire tags through \
                                       normal_text",
                                dropped_block = %truncate_for_log(block),
                                "GLM-4.7 parser dropping truncated tool_call block (recovery attempt failed)"
                            );
                        }
                    }
                } else {
                    // Either recovery disabled (production default for GLM-4.7)
                    // or no <arg_key> in the tail (so this is plausibly not a
                    // real tool call at all, just a stray <tool_call> token).
                    let reason = if !config.allow_eof_recovery {
                        "allow_eof_recovery=false (production default for GLM-4.7 to match \
                         vLLM/SGLang on truncated tool calls)"
                    } else {
                        "no <arg_key> in the tail after the <tool_call> start fence, so the \
                         block does not look like a structurally-real GLM-4.7 tool call"
                    };
                    warn!(
                        why = %reason,
                        dropped_block = %truncate_for_log(block),
                        "GLM-4.7 parser dropping truncated tool_call block (no end fence)"
                    );
                }
                // Drop the truncated/unrecoverable tail. Emitting the raw
                // <tool_call>...<arg_key>...<arg_value>... prefix as
                // normal_text would leak wire tags into message.content; vLLM
                // strips the same way on truncation.
                break;
            }
        } else {
            // No more tool calls
            let gap = &text[cursor..];
            if let Some((prefix, mut parsed_calls)) =
                recover_bare_glm47_calls_in_span(gap, config, tools)?
            {
                if calls.is_empty() {
                    normal_parts.push(prefix);
                }
                calls.append(&mut parsed_calls);
            } else if calls.is_empty() {
                if let Some(marker_idx) = first_orphan_glm47_marker_index(gap, config) {
                    normal_parts.push(orphan_glm47_prefix(gap, marker_idx));
                } else {
                    normal_parts.push(gap.to_string());
                }
            }
            break;
        }
    }

    let normal_text = normal_parts.join("");
    let normal_text = if calls.is_empty() {
        normal_text.trim().to_string()
    } else {
        normal_text
    };
    Ok((normal_text, calls))
}

fn recover_bare_glm47_calls_in_span(
    span: &str,
    config: &Glm47ParserConfig,
    tools: Option<&[ToolDefinition]>,
) -> anyhow::Result<Option<(String, Vec<ToolCallResponse>)>> {
    let Some(marker_idx) = first_orphan_glm47_marker_index(span, config) else {
        return Ok(None);
    };
    let recovered = recover_bare_glm47_calls(span, marker_idx, config, tools)?;
    if let Some((prefix, parsed_calls)) = recovered {
        let recovered_calls = parsed_calls.len();
        warn!(
            why = "bare_body_gap_recovery",
            recovered_calls,
            recovered_bytes = span.len() - prefix.len(),
            kept_prefix_bytes = prefix.len(),
            "GLM-4.7 parser recovered complete bare call body/bodies before a later <tool_call>"
        );
        return Ok(Some((prefix, parsed_calls)));
    }
    Ok(None)
}

fn first_orphan_glm47_marker_index(text: &str, config: &Glm47ParserConfig) -> Option<usize> {
    [
        config.tool_call_end.as_str(),
        config.arg_key_start.as_str(),
        config.arg_key_end.as_str(),
        config.arg_value_start.as_str(),
        config.arg_value_end.as_str(),
    ]
    .into_iter()
    .filter_map(|marker| text.find(marker))
    .min()
}

fn orphan_glm47_prefix(text: &str, marker_idx: usize) -> String {
    let prefix = text[..marker_idx].trim_end();
    let token_start = prefix
        .char_indices()
        .rev()
        .find(|(_, ch)| ch.is_whitespace())
        .map(|(idx, ch)| idx + ch.len_utf8())
        .unwrap_or(0);
    let tail = &prefix[token_start..];
    if !tail.is_empty()
        && tail
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '.'))
    {
        prefix[..token_start].trim().to_string()
    } else {
        prefix.trim().to_string()
    }
}

fn recover_bare_glm47_calls(
    text: &str,
    marker_idx: usize,
    config: &Glm47ParserConfig,
    tools: Option<&[ToolDefinition]>,
) -> anyhow::Result<Option<(String, Vec<ToolCallResponse>)>> {
    if !text[marker_idx..].contains(config.tool_call_end.as_str()) {
        return Ok(None);
    }

    let before_marker = text[..marker_idx].trim_end();
    let function_name_start = before_marker
        .char_indices()
        .rev()
        .find(|(_, ch)| ch.is_whitespace())
        .map(|(idx, ch)| idx + ch.len_utf8())
        .unwrap_or(0);

    let candidate_name = before_marker[function_name_start..].trim();
    if candidate_name.is_empty()
        || !candidate_name
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '.'))
    {
        return Ok(None);
    }

    let prefix = text[..function_name_start].to_string();
    let mut cursor = function_name_start;
    let mut calls = Vec::new();

    while cursor < text.len() {
        let rest = &text[cursor..];
        let trim_offset = rest.len() - rest.trim_start().len();
        cursor += trim_offset;

        let tail = &text[cursor..];
        let Some(end_pos) = tail.find(config.tool_call_end.as_str()) else {
            break;
        };
        let call_end = cursor + end_pos + config.tool_call_end.len();
        let wrapped = format!("{}{}", config.tool_call_start, &text[cursor..call_end]);
        calls.push(parse_tool_call_block(&wrapped, config, tools)?);
        cursor = call_end;

        if is_glm47_close_marker_spam(&text[cursor..], config) {
            warn!(
                why = "orphan_close_marker_spam",
                dropped_block = %truncate_for_log(&text[cursor..]),
                "GLM-4.7 parser dropping orphan close-marker spam after recovered bare call"
            );
            break;
        }

        if first_orphan_glm47_marker_index(&text[cursor..], config).is_none() {
            break;
        }
    }

    if calls.is_empty() {
        return Ok(None);
    }
    Ok(Some((prefix, calls)))
}

fn is_glm47_close_marker_spam(text: &str, config: &Glm47ParserConfig) -> bool {
    let mut rest = text.trim_start();
    let mut saw_close = false;
    while let Some(after_close) = rest.strip_prefix(config.tool_call_end.as_str()) {
        saw_close = true;
        rest = after_close.trim_start();
    }
    saw_close && rest.is_empty()
}

fn consume_glm47_close_markers(text: &str, mut cursor: usize, config: &Glm47ParserConfig) -> usize {
    loop {
        let rest = &text[cursor..];
        let trim_offset = rest.len() - rest.trim_start().len();
        let close_start = cursor + trim_offset;
        if !text[close_start..].starts_with(config.tool_call_end.as_str()) {
            return cursor;
        }
        cursor = close_start + config.tool_call_end.len();
    }
}

/// Coerce a raw string value using the tool's parameter schema.
/// Falls back to string if no schema is available or the type is unrecognized.
fn coerce_value(raw: &str, schema_type: Option<&str>) -> ParsedValue {
    let trimmed = raw.trim();

    // A `string` parameter is delivered verbatim: the model's text may
    // legitimately look like JSON (an object, array, or quoted text), and
    // parsing it would change its type behind the schema's back.
    if matches!(schema_type, Some("string")) {
        return Value::String(raw.to_string()).into();
    }

    // If the value already looks like JSON (object, array, or quoted string), parse it directly
    if (trimmed.starts_with('{') || trimmed.starts_with('[') || trimmed.starts_with('"'))
        && let Ok(v) = serde_json::from_str::<Value>(trimmed)
    {
        return v.into();
    }

    // Use schema type hints for coercion when available
    match schema_type {
        Some("integer") | Some("int") => {
            if let Some(value) = coerce_integer_literal(trimmed) {
                return value;
            }
        }
        Some("number") | Some("float") | Some("double") => {
            if let Some(value) = coerce_integer_literal(trimmed) {
                return value;
            }
            if let Ok(n) = trimmed.parse::<f64>()
                && let Some(num) = serde_json::Number::from_f64(n)
            {
                return Value::Number(num).into();
            }
        }
        Some("boolean") | Some("bool") => match trimmed.to_lowercase().as_str() {
            "true" | "1" | "yes" => return Value::Bool(true).into(),
            "false" | "0" | "no" => return Value::Bool(false).into(),
            _ => {}
        },
        Some("array") => {
            // Try JSON parse first, then fall back to comma-separated splitting
            if let Ok(v) = serde_json::from_str::<Value>(trimmed)
                && v.is_array()
            {
                return v.into();
            }
            let items: Vec<Value> = trimmed
                .split(',')
                .map(|s| Value::String(s.trim().to_string()))
                .collect();
            return Value::Array(items).into();
        }
        Some("null") if trimmed == "null" || trimmed == "None" || trimmed.is_empty() => {
            return Value::Null.into();
        }
        _ => {}
    }

    Value::String(raw.to_string()).into()
}

/// The type hint for an argument and the text to coerce: the raw text, or, for a
/// quoted JSON string under a schema that admits no string but does admit the
/// scalar type the quoted text parses as, the unquoted text.
struct ArgumentHint<'a> {
    schema_type: Option<&'static str>,
    text: std::borrow::Cow<'a, str>,
}

/// Look up the JSON Schema type for a parameter by name from a tool's parameter schema.
///
/// The parameter's schema is found through the root's composition keywords
/// (`anyOf`/`oneOf`/`allOf`, `if`/`then`/`else`, local `$ref`), not only a
/// top-level `properties`. When the raw text parses as a JSON scalar and the
/// schema admits that type, the scalar wins over a bare `string` branch; a
/// string `const`/`enum` member that equals the text keeps it as a string. A
/// schema with no type information at all takes the type the text parses as.
fn resolve_argument<'a>(
    tools: Option<&[ToolDefinition]>,
    function_name: &str,
    param_name: &str,
    raw: &'a str,
) -> ArgumentHint<'a> {
    let verbatim = |schema_type| ArgumentHint {
        schema_type,
        text: std::borrow::Cow::Borrowed(raw),
    };
    let Some(schema) = tools
        .and_then(|tools| tools.iter().find(|t| t.name == function_name))
        .and_then(|tool| tool.parameters.as_ref())
    else {
        return verbatim(None);
    };
    let mut found: Vec<&Value> = Vec::new();
    let mut remaining = MAX_PARAM_LOOKUP_NODES;
    find_param_schemas(schema, schema, param_name, 0, &mut remaining, &mut found);
    // Several branches may each declare the parameter: treat them as a union.
    let union: Value;
    let param: &Value = match found.as_slice() {
        [] => return verbatim(None),
        [single] => single,
        many => {
            union = serde_json::json!({
                "anyOf": many.iter().map(|v| (*v).clone()).collect::<Vec<Value>>()
            });
            &union
        }
    };
    let schema_type = get_param_schema_type_for(schema, param, raw);
    if raw.trim().starts_with('"')
        && let Ok(Value::String(inner)) = serde_json::from_str::<Value>(raw.trim())
        && schema_type_hint(schema, param, "string", raw) == Some(false)
        && let Some(inner_type) = get_param_schema_type_for(schema, param, &inner)
        && !matches!(inner_type, "string" | "array" | "object")
    {
        return ArgumentHint {
            schema_type: Some(inner_type),
            text: std::borrow::Cow::Owned(inner),
        };
    }
    verbatim(schema_type)
}

const MAX_PARAM_LOOKUP_NODES: usize = 4096;

/// Collect every schema the root assigns to `param_name`, following the
/// composition keywords a tool schema may wrap its object in.
fn find_param_schemas<'a>(
    root: &'a Value,
    node: &'a Value,
    param_name: &str,
    depth: usize,
    remaining: &mut usize,
    out: &mut Vec<&'a Value>,
) {
    if depth > 16 || *remaining == 0 {
        return;
    }
    *remaining -= 1;
    if node.get("$dynamicRef").is_some()
        || node.get("$recursiveRef").is_some()
        || (node.get("$id").is_some() && !std::ptr::eq(node, root))
    {
        return;
    }
    if let Some(param) = node
        .get("properties")
        .and_then(|props| props.get(param_name))
        && !out.iter().any(|seen| std::ptr::eq(*seen, param))
    {
        out.push(param);
    }
    if let Some(target) = node
        .get("$ref")
        .and_then(Value::as_str)
        .and_then(|reference| resolve_local_schema_ref(reference, root))
    {
        find_param_schemas(root, target, param_name, depth + 1, remaining, out);
    }
    for keyword in ["anyOf", "oneOf", "allOf"] {
        if let Some(options) = node.get(keyword).and_then(Value::as_array) {
            for option in options {
                find_param_schemas(root, option, param_name, depth + 1, remaining, out);
            }
        }
    }
    for keyword in ["if", "then", "else"] {
        if let Some(sub) = node.get(keyword) {
            find_param_schemas(root, sub, param_name, depth + 1, remaining, out);
        }
    }
}

fn static_type_name(name: &str) -> Option<&'static str> {
    Some(match name {
        "string" => "string",
        "integer" => "integer",
        "int" => "int",
        "number" => "number",
        "float" => "float",
        "double" => "double",
        "boolean" => "boolean",
        "bool" => "bool",
        "array" => "array",
        "object" => "object",
        "null" => "null",
        _ => return None,
    })
}

fn get_param_schema_type_for(schema: &Value, param: &Value, raw: &str) -> Option<&'static str> {
    // Prefer JSON null for a bare null when the schema permits it. The wire
    // spelling can also represent a string; keep string preference for other values.
    if raw.trim() == "null" {
        return Some(if schema_permits_null(param, schema) {
            "null"
        } else {
            "string"
        });
    }
    // A string constant or enum member equal to the text claims it verbatim.
    if schema_string_literal_claims(schema, param, raw) {
        return Some("string");
    }
    // Candidate scalar types from the JSON value, not branch order.
    let trimmed = raw.trim();
    let candidates: &[&str] = if !matches!(
        trimmed.as_bytes().first(),
        Some(b'n' | b't' | b'f' | b'-' | b'0'..=b'9')
    ) {
        &[]
    } else if super::parsed_value::is_integer_literal(trimmed) {
        // Preserve the integer coercer's arbitrary-length path before Value's numeric limit.
        &["integer", "number"]
    } else {
        match serde_json::from_str::<Value>(trimmed).ok() {
            Some(Value::Null) => &["null"],
            Some(Value::Bool(_)) => &["boolean"],
            Some(Value::Number(_)) => &["number"],
            _ => &[],
        }
    };
    let mut all_unknown = true;
    for candidate in candidates {
        match schema_type_hint(schema, param, candidate, raw) {
            Some(true) => return Some(candidate),
            // An integral value a schema rejects as an integer is rejected as a number too.
            Some(false) => {
                all_unknown = false;
                if *candidate == "integer" {
                    break;
                }
            }
            None => {}
        }
    }
    match schema_type_hint(schema, param, "string", raw) {
        Some(true) => return Some("string"),
        Some(false) => all_unknown = false,
        None => {}
    }
    if let Some(schema_type) = param.get("type").and_then(Value::as_str) {
        return static_type_name(schema_type);
    }
    // A schema that resolves fully yet names no type takes what the text parses as.
    // Unresolvable, cyclic or scoped references stay conservative: verbatim text.
    if all_unknown && schema_fully_resolvable(schema, param) {
        return candidates.first().copied();
    }
    None
}

fn schema_fully_resolvable(root: &Value, schema: &Value) -> bool {
    let mut remaining = 1024;
    let mut active: Vec<&Value> = Vec::new();
    schema_resolvable(root, schema, 0, &mut remaining, &mut active)
}

fn schema_resolvable<'a>(
    root: &'a Value,
    schema: &'a Value,
    depth: usize,
    remaining: &mut usize,
    active: &mut Vec<&'a Value>,
) -> bool {
    if depth > 16 || *remaining == 0 {
        return false;
    }
    *remaining -= 1;
    if schema.get("$dynamicRef").is_some()
        || schema.get("$recursiveRef").is_some()
        || (schema.get("$id").is_some() && !std::ptr::eq(schema, root))
    {
        return false;
    }
    if let Some(reference) = schema.get("$ref") {
        let Some(target) = reference
            .as_str()
            .and_then(|reference| resolve_local_schema_ref(reference, root))
        else {
            return false;
        };
        if active.iter().any(|seen| std::ptr::eq(*seen, target)) {
            return false;
        }
        active.push(target);
        let ok = schema_resolvable(root, target, depth + 1, remaining, active);
        active.pop();
        if !ok {
            return false;
        }
    }
    for keyword in ["anyOf", "oneOf", "allOf"] {
        if let Some(options) = schema.get(keyword).and_then(Value::as_array)
            && !options
                .iter()
                .all(|option| schema_resolvable(root, option, depth + 1, remaining, active))
        {
            return false;
        }
    }
    ["not", "if", "then", "else"].iter().all(|keyword| {
        schema
            .get(keyword)
            .is_none_or(|sub| schema_resolvable(root, sub, depth + 1, remaining, active))
    })
}

fn schema_type_hint(root: &Value, schema: &Value, expected: &str, raw: &str) -> Option<bool> {
    let mut remaining = 1024;
    let matched = schema_type_match(root, schema, expected, raw, 0, &mut remaining);
    (remaining > 0).then_some(matched).flatten()
}

/// Whether a string `const` or `enum` member reachable from `schema` equals `raw`.
fn schema_string_literal_claims(root: &Value, schema: &Value, raw: &str) -> bool {
    let mut remaining = 1024;
    string_literal_claims(root, schema, raw, 0, &mut remaining)
}

fn string_literal_claims(
    root: &Value,
    schema: &Value,
    raw: &str,
    depth: usize,
    remaining: &mut usize,
) -> bool {
    if depth > 16 || *remaining == 0 || schema.get("$id").is_some() && !std::ptr::eq(schema, root) {
        return false;
    }
    *remaining -= 1;
    if schema.get("const").and_then(Value::as_str) == Some(raw) {
        return true;
    }
    if schema
        .get("enum")
        .and_then(Value::as_array)
        .is_some_and(|members| members.iter().any(|m| m.as_str() == Some(raw)))
    {
        return true;
    }
    if schema
        .get("$ref")
        .and_then(Value::as_str)
        .and_then(|reference| resolve_local_schema_ref(reference, root))
        .is_some_and(|target| string_literal_claims(root, target, raw, depth + 1, remaining))
    {
        return true;
    }
    ["anyOf", "oneOf", "allOf"].iter().any(|keyword| {
        schema
            .get(keyword)
            .and_then(Value::as_array)
            .is_some_and(|options| {
                options
                    .iter()
                    .any(|option| string_literal_claims(root, option, raw, depth + 1, remaining))
            })
    })
}

const MAX_NULL_SCHEMA_REF_DEPTH: usize = 16;
const MAX_NULL_SCHEMA_NODES: usize = 4096;

// Only a proven match may turn the model's text into JSON null. Unknown references
// must remain unknown through negation and oneOf, rather than counting as false.
fn schema_permits_null(schema: &Value, root: &Value) -> bool {
    let mut active_refs = vec![schema];
    let mut remaining = MAX_NULL_SCHEMA_NODES;
    schema_null_match(schema, root, &mut active_refs, 0, &mut remaining) == Some(true)
}

fn intersect_null_matches(left: Option<bool>, right: Option<bool>) -> Option<bool> {
    match (left, right) {
        (Some(false), _) | (_, Some(false)) => Some(false),
        (Some(true), Some(true)) => Some(true),
        _ => None,
    }
}

pub(crate) fn has_unsupported_schema_ref_scope(schema: &Value) -> bool {
    ["$id", "$dynamicRef", "$recursiveRef"]
        .iter()
        .any(|keyword| schema.get(*keyword).and_then(Value::as_str).is_some())
}

pub(crate) fn resolve_local_schema_ref<'a>(reference: &str, root: &'a Value) -> Option<&'a Value> {
    // URI percent-decoding precedes JSON Pointer's ~0/~1 decoding.
    let pointer = reference.strip_prefix('#')?;
    let decoded;
    let pointer = if pointer.contains('%') {
        let mut bytes = pointer.bytes();
        let mut result = Vec::with_capacity(pointer.len());
        while let Some(byte) = bytes.next() {
            result.push(if byte == b'%' {
                let high = char::from(bytes.next()?).to_digit(16)?;
                let low = char::from(bytes.next()?).to_digit(16)?;
                ((high << 4) | low) as u8
            } else {
                byte
            });
        }
        decoded = String::from_utf8(result).ok()?;
        decoded.as_str()
    } else {
        pointer
    };
    let mut bytes = pointer.bytes();
    while let Some(byte) = bytes.next() {
        if byte == b'~' && !matches!(bytes.next(), Some(b'0' | b'1')) {
            return None;
        }
    }
    // A nested $id changes the reference scope. Do not jump through it while
    // resolving a fragment against the original tool-parameter document.
    for (end, _) in pointer.char_indices().filter(|(_, ch)| *ch == '/').skip(1) {
        if has_unsupported_schema_ref_scope(root.pointer(&pointer[..end])?) {
            return None;
        }
    }
    let target = root.pointer(pointer)?;
    matches!(target, Value::Bool(_) | Value::Object(_)).then_some(target)
}

// Evaluate only the constraints relevant to null. Ref targets and sibling
// keywords intersect; the remaining coercion rules do not change.
fn schema_null_match<'a>(
    schema: &'a Value,
    root: &'a Value,
    active_refs: &mut Vec<&'a Value>,
    ref_depth: usize,
    remaining: &mut usize,
) -> Option<bool> {
    *remaining = (*remaining).checked_sub(1)?;
    if let Some(allowed) = schema.as_bool() {
        return Some(allowed);
    }
    if !schema.is_object()
        || schema.get("$dynamicRef").is_some()
        || schema.get("$recursiveRef").is_some()
        || (schema.get("$id").is_some() && !std::ptr::eq(schema, root))
    {
        return None;
    }
    let mut permits = Some(true);
    if let Some(reference) = schema.get("$ref") {
        let target = reference
            .as_str()
            .and_then(|reference| resolve_local_schema_ref(reference, root));
        permits = if let Some(target) = target
            && ref_depth < MAX_NULL_SCHEMA_REF_DEPTH
            && !active_refs
                .iter()
                .any(|active| std::ptr::eq(*active, target))
        {
            active_refs.push(target);
            let result = schema_null_match(target, root, active_refs, ref_depth + 1, remaining);
            active_refs.pop();
            result
        } else {
            None
        };
    }
    if let Some(ty) = schema.get("type") {
        let nullable = schema.get("nullable").and_then(Value::as_bool) == Some(true);
        if !nullable
            && ty.as_str() != Some("null")
            && !ty
                .as_array()
                .is_some_and(|types| types.iter().any(|ty| ty == "null"))
        {
            return Some(false);
        }
    }
    if schema.get("const").is_some_and(|value| !value.is_null())
        || schema
            .get("enum")
            .and_then(Value::as_array)
            .is_some_and(|values| !values.iter().any(Value::is_null))
    {
        return Some(false);
    }
    for keyword in ["allOf", "anyOf", "oneOf"] {
        if let Some(branches) = schema.get(keyword).and_then(Value::as_array) {
            let mut matches = 0;
            let mut unknown = 0;
            for branch in branches {
                match schema_null_match(branch, root, active_refs, ref_depth, remaining) {
                    Some(true) => matches += 1,
                    Some(false) => {}
                    None => unknown += 1,
                }
            }
            let branch_match = match keyword {
                "allOf" if matches + unknown < branches.len() => Some(false),
                "anyOf" if matches > 0 => Some(true),
                "oneOf" if matches > 1 => Some(false),
                _ if unknown > 0 => None,
                "allOf" => Some(true),
                "anyOf" => Some(false),
                _ => Some(matches == 1),
            };
            permits = intersect_null_matches(permits, branch_match);
            if permits == Some(false) {
                return permits;
            }
        }
    }
    if let Some(not) = schema.get("not") {
        let not_match = schema_null_match(not, root, active_refs, ref_depth, remaining);
        permits = intersect_null_matches(permits, not_match.map(|matches| !matches));
    }
    permits
}

fn schema_has_type(root: &Value, schema: &Value, expected: &str, raw: &str) -> bool {
    let mut remaining = 1024;
    let matched = schema_type_match(root, schema, expected, raw, 0, &mut remaining);
    remaining > 0 && matched == Some(true)
}

// None means no type hint: e.g. minLength alone must not exclude an allOf sibling's type.
fn schema_type_match(
    root: &Value,
    schema: &Value,
    expected: &str,
    raw: &str,
    depth: usize,
    remaining: &mut usize,
) -> Option<bool> {
    // A branching reference cycle can expand exponentially even at bounded depth.
    *remaining = remaining.checked_sub(1)?;
    // Local references are common in strict tool schemas. Limit traversal so a
    // cyclic definition cannot recurse indefinitely while deciding a type hint.
    if depth > 16
        || schema.get("$dynamicRef").is_some()
        || schema.get("$recursiveRef").is_some()
        || (schema.get("$id").is_some() && !std::ptr::eq(schema, root))
    {
        return None;
    }
    let reference_hint = schema
        .get("$ref")
        .and_then(Value::as_str)
        .and_then(|reference| resolve_local_schema_ref(reference, root))
        .and_then(|target| schema_type_match(root, target, expected, raw, depth + 1, remaining));
    let matches = |ty: &Value| {
        ty.as_str() == Some(expected) || (expected == "integer" && ty.as_str() == Some("number"))
    };
    let mut hint = schema.get("type").map(|ty| {
        ty.as_array()
            .map_or_else(|| matches(ty), |types| types.iter().any(matches))
    });
    // A string constant preserves literal XML text only when this value
    // matches it. Otherwise a union's numeric/structured alternative must
    // remain available. Intersect with types, refs, and composition siblings.
    if expected == "string"
        && let Some(value) = schema.get("const")
    {
        hint = Some(hint.unwrap_or(true) && value.as_str() == Some(raw));
    }
    // Modern JSON Schema applies $ref siblings as additional constraints.
    hint = match (hint, reference_hint) {
        (Some(left), Some(right)) => Some(left && right),
        (left, right) => left.or(right),
    };
    // enum/const-derived type hint: a parameter declared only by its allowed values
    // (`{"enum": [1, 2, 4]}`, `{"const": true}`) admits the JSON types of those
    // values; a declared `type` keeps its hint unchanged. Integral numbers count
    // as integer and number. A string member matches only this exact value, so a
    // union's numeric/structured alternative stays reachable for other text.
    let value_matches = |value: &Value| -> bool {
        match value {
            Value::Null => expected == "null",
            Value::Bool(_) => expected == "boolean",
            Value::Number(n) => {
                expected == "number"
                    || (expected == "integer" && n.as_f64().is_some_and(|f| f.fract() == 0.0))
            }
            Value::String(member) => expected == "string" && member == raw,
            Value::Array(_) => expected == "array",
            Value::Object(_) => expected == "object",
        }
    };
    let literal_hint = match (
        schema.get("type"),
        schema.get("enum").and_then(Value::as_array),
        schema.get("const"),
    ) {
        (Some(_), _, _) => None,
        (None, Some(values), _) if !values.is_empty() => Some(values.iter().any(value_matches)),
        (None, _, Some(value)) => Some(value_matches(value)),
        _ => None,
    };
    hint = match (hint, literal_hint) {
        (Some(left), Some(right)) => Some(left && right),
        (left, right) => left.or(right),
    };
    for keyword in ["anyOf", "oneOf", "allOf"] {
        let Some(options) = schema.get(keyword).and_then(Value::as_array) else {
            continue;
        };
        let branches = options
            .iter()
            .map(|option| schema_type_match(root, option, expected, raw, depth, remaining));
        let branch_hint = if keyword == "allOf" {
            branches.flatten().reduce(|left, right| left && right)
        } else {
            branches
                .reduce(|left, right| match (left, right) {
                    (Some(true), _) | (_, Some(true)) => Some(true),
                    (Some(false), Some(false)) => Some(false),
                    _ => None,
                })
                .flatten()
        };
        hint = match (hint, branch_hint) {
            (Some(left), Some(right)) => Some(left && right),
            (left, right) => left.or(right),
        };
    }
    // `not`: a type the negated schema proves is excluded; one it proves
    // absent is admitted. An unknown inner hint stays unknown.
    if let Some(negated) = schema.get("not") {
        let not_hint =
            schema_type_match(root, negated, expected, raw, depth + 1, remaining).map(|b| !b);
        hint = match (hint, not_hint) {
            (Some(left), Some(right)) => Some(left && right),
            (left, right) => left.or(right),
        };
    }
    // `if`/`then`/`else`: the value may satisfy `if` and `then`, or fail `if`
    // and satisfy `else`; the hint is the union of the two paths.
    if let Some(condition) = schema.get("if") {
        let cond = schema_type_match(root, condition, expected, raw, depth + 1, remaining);
        let then_hint = schema
            .get("then")
            .map(|sub| schema_type_match(root, sub, expected, raw, depth + 1, remaining));
        let else_hint = schema
            .get("else")
            .map(|sub| schema_type_match(root, sub, expected, raw, depth + 1, remaining));
        let and = |left: Option<bool>, right: Option<Option<bool>>| match (left, right) {
            (Some(l), Some(Some(r))) => Some(l && r),
            (l, Some(r)) => l.or(r),
            (l, None) => l,
        };
        let when_true = and(cond, then_hint);
        let when_false = and(cond.map(|c| !c), else_hint);
        let branch_hint = match (when_true, when_false) {
            (Some(true), _) | (_, Some(true)) => Some(true),
            (Some(false), Some(false)) => Some(false),
            _ => None,
        };
        hint = match (hint, branch_hint) {
            (Some(left), Some(right)) => Some(left && right),
            (left, right) => left.or(right),
        };
    }
    hint
}

/// A top-level `required` key the model left out is present as JSON null when its
/// schema admits null: the only value that satisfies such a schema without the key.
fn fill_required_nullable(schema: &Value, arguments: &mut Vec<(String, ParsedValue)>) {
    let Some(required) = schema.get("required").and_then(Value::as_array) else {
        return;
    };
    for name in required.iter().filter_map(Value::as_str) {
        if arguments.iter().any(|(key, _)| key == name) {
            continue;
        }
        let mut found: Vec<&Value> = Vec::new();
        let mut remaining = MAX_PARAM_LOOKUP_NODES;
        find_param_schemas(schema, schema, name, 0, &mut remaining, &mut found);
        if found.iter().any(|param| schema_permits_null(param, schema)) {
            arguments.push((name.to_string(), Value::Null.into()));
        }
    }
}

/// Parse a single GLM-4.7 tool call block
/// Format: <tool_call>function_name<arg_key>key1</arg_key><arg_value>value1</arg_value>...</tool_call>
fn parse_tool_call_block(
    block: &str,
    config: &Glm47ParserConfig,
    tools: Option<&[ToolDefinition]>,
) -> anyhow::Result<ToolCallResponse> {
    // Remove the outer <tool_call> tags
    let start_token = &config.tool_call_start;
    let end_token = &config.tool_call_end;

    // Strip the outer start token. The end token is optional so we can
    // recover from max_tokens / EOS truncation that drops `</tool_call>`.
    let after_start = block
        .strip_prefix(start_token.as_str())
        .ok_or_else(|| anyhow::anyhow!("Invalid tool call block format"))?;
    let content = after_start
        .strip_suffix(end_token.as_str())
        .unwrap_or(after_start);

    // Extract function name (everything before first <arg_key> or end)
    let arg_key_start = &config.arg_key_start;
    let function_name = if let Some(pos) = content.find(arg_key_start.as_str()) {
        content[..pos].trim().to_string()
    } else {
        // No arguments, just function name
        content.trim().to_string()
    };

    if function_name.is_empty() {
        anyhow::bail!("Empty function name in tool call");
    }

    // Parse key-value pairs, keeping the order the model emitted them in.
    let mut arguments: Vec<(String, ParsedValue)> = Vec::new();
    let mut argument_indices: HashMap<&str, usize> = HashMap::new();
    let args_section = &content[function_name.len()..];

    // Build regex patterns
    let arg_key_start_escaped = regex::escape(&config.arg_key_start);
    let arg_key_end_escaped = regex::escape(&config.arg_key_end);
    let arg_value_start_escaped = regex::escape(&config.arg_value_start);
    let arg_value_end_escaped = regex::escape(&config.arg_value_end);

    // Pattern to match: <arg_key>key</arg_key><arg_value>value</arg_value>
    // (?s) enables dotall mode so (.*?) matches across newlines — required
    // because models often emit multi-line content in arg values.
    let pattern = format!(
        r"(?s){}([^<]+){}\s*{}(.*?){}",
        arg_key_start_escaped, arg_key_end_escaped, arg_value_start_escaped, arg_value_end_escaped
    );

    let regex = Regex::new(&pattern)?;

    for cap in regex.captures_iter(args_section) {
        let key = cap.get(1).map(|m| m.as_str().trim()).unwrap_or("");
        let raw_value = cap.get(2).map(|m| m.as_str()).unwrap_or("");

        if !key.is_empty() {
            // Look up the expected type from the tool's parameter schema
            let hint = resolve_argument(tools, &function_name, key, raw_value);
            let json_value = coerce_value(&hint.text, hint.schema_type);

            match argument_indices.get(key).copied() {
                Some(index) => arguments[index].1 = json_value,
                None => {
                    argument_indices.insert(key, arguments.len());
                    arguments.push((key.to_string(), json_value));
                }
            }
        }
    }

    if let Some(schema) = tools
        .and_then(|tools| tools.iter().find(|t| t.name == function_name))
        .and_then(|tool| tool.parameters.as_ref())
    {
        fill_required_nullable(schema, &mut arguments);
    }

    // Preserve unknown calls; executors must authorize names against request-scoped tools.
    if let Some(tools_list) = tools
        && !tools_list.iter().any(|t| t.name == function_name)
    {
        warn!(
            function = %function_name,
            why = "tool_not_in_request_tool_list",
            "GLM-4.7 tool call references a function not in the request's tools list; passing it through"
        );
    }

    Ok(ToolCallResponse {
        id: Uuid::new_v4().to_string(),
        tp: ToolCallType::Function,
        function: CalledFunction {
            name: function_name,
            arguments: serde_json::to_string(&OrderedArguments(&arguments))?,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn get_test_config() -> Glm47ParserConfig {
        Glm47ParserConfig::default()
    }

    fn parse_schema_value(parameter: Value, raw: &str) -> Value {
        let tools = vec![ToolDefinition {
            name: "probe".into(),
            parameters: Some(serde_json::json!({
                "type": "object", "properties": {"value": parameter}
            })),
            strict: None,
        }];
        let wire = format!(
            "<tool_call>probe<arg_key>value</arg_key><arg_value>{raw}</arg_value></tool_call>"
        );
        let (calls, _) =
            try_tool_call_parse_glm47(&wire, &get_test_config(), Some(&tools)).unwrap();
        assert_eq!(calls.len(), 1);
        let arguments: Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        arguments["value"].clone()
    }

    #[test]
    fn string_constant_unions_select_matching_values() {
        for keyword in ["anyOf", "oneOf"] {
            for reverse in [false, true] {
                for (kind, raw, expected) in [
                    ("integer", "42", serde_json::json!(42)),
                    ("integer", "-7", serde_json::json!(-7)),
                    ("number", "3.5", serde_json::json!(3.5)),
                    ("boolean", "true", serde_json::json!(true)),
                    ("object", "{\"x\":1}", serde_json::json!({"x": 1})),
                    ("array", "[1,2]", serde_json::json!([1, 2])),
                    ("null", "null", serde_json::json!(null)),
                ] {
                    let mut branches = vec![
                        serde_json::json!({"const": "auto"}),
                        serde_json::json!({"type": kind}),
                    ];
                    if reverse {
                        branches.reverse();
                    }
                    let parameter = serde_json::json!({keyword: branches});
                    assert_eq!(
                        parse_schema_value(parameter.clone(), raw),
                        expected,
                        "{keyword}, reverse={reverse}, kind={kind}, raw={raw}"
                    );
                    assert_eq!(
                        parse_schema_value(parameter, "auto"),
                        serde_json::json!("auto"),
                        "{keyword}, reverse={reverse}, kind={kind}, string branch"
                    );
                }
            }
        }
    }

    #[test]
    fn string_constants_preserve_literal_text() {
        for raw in [
            "auto",
            "\"hello\"",
            "{\"x\":1}",
            "[1,2]",
            "42",
            "\n42\t",
            "null",
        ] {
            for parameter in [
                serde_json::json!({"const": raw}),
                serde_json::json!({"anyOf": [{"const": raw}, {"type": "integer"}]}),
                serde_json::json!({"allOf": [{"const": raw}, {"minLength": 1}]}),
            ] {
                assert_eq!(
                    parse_schema_value(parameter.clone(), raw),
                    serde_json::json!(raw),
                    "schema={parameter}, raw={raw:?}"
                );
            }
        }
        assert_eq!(
            parse_schema_value(
                serde_json::json!({"anyOf": [{"const": "42"}, {"type": "integer"}]}),
                " 42 "
            ),
            serde_json::json!(42),
            "a string constant must match raw text, not its trimmed spelling"
        );
    }

    #[test]
    fn string_constant_hints_follow_refs_and_intersect_constraints() {
        let schema = serde_json::json!({
            "$defs": {
                "Literal": {"const": "\"hello\""},
                "Alias": {"$ref": "#/$defs/Literal"},
                "a/b~c": {"const": "\"hello\""},
                "Integer": {"type": "integer"},
                "Choice": {"anyOf": [{"const": "auto"}, {"type": "integer"}]}
            }
        });
        for (parameter, raw, expected) in [
            (
                serde_json::json!({"$ref": "#/$defs/Alias"}),
                "\"hello\"",
                true,
            ),
            (
                serde_json::json!({"$ref": "#/$defs/a~1b~0c"}),
                "\"hello\"",
                true,
            ),
            (
                serde_json::json!({"$ref": "#/$defs/Alias", "type": "integer"}),
                "\"hello\"",
                false,
            ),
            (
                serde_json::json!({"$ref": "#/$defs/Integer", "const": "42"}),
                "42",
                false,
            ),
            (
                serde_json::json!({"type": "integer", "const": "42"}),
                "42",
                false,
            ),
            (
                serde_json::json!({"allOf": [{"const": "42"}, {"type": "integer"}]}),
                "42",
                false,
            ),
            (
                serde_json::json!({"allOf": [{"$ref": "#/$defs/Literal"}, {"type": "string"}]}),
                "\"hello\"",
                true,
            ),
            (
                serde_json::json!({"allOf": [{"$ref": "#/$defs/Choice"}, {"minimum": 0}]}),
                "42",
                false,
            ),
            (serde_json::json!({"$ref": "#/$defs/Choice"}), "auto", true),
            (serde_json::json!({"$ref": "#/$defs/Choice"}), "42", false),
        ] {
            assert_eq!(
                schema_has_type(&schema, &parameter, "string", raw),
                expected,
                "schema={parameter}, raw={raw:?}"
            );
        }
    }

    #[test]
    fn string_constant_unions_keep_existing_ambiguity_rules() {
        for (parameter, raw, expected) in [
            (
                serde_json::json!({"anyOf": [{"const": "42"}, {"type": "integer"}]}),
                "42",
                serde_json::json!("42"),
            ),
            (
                serde_json::json!({"anyOf": [{"const": "null"}, {"type": "null"}]}),
                "null",
                serde_json::json!(null),
            ),
            (
                serde_json::json!({"const": "null"}),
                "null",
                serde_json::json!("null"),
            ),
            (
                serde_json::json!({"type": ["string", "null"], "const": "null"}),
                "null",
                serde_json::json!("null"),
            ),
            (
                serde_json::json!({"type": "string", "enum": ["\"hello\""]}),
                "\"hello\"",
                serde_json::json!("\"hello\""),
            ),
        ] {
            assert_eq!(
                parse_schema_value(parameter.clone(), raw),
                expected,
                "schema={parameter}, raw={raw:?}"
            );
        }
    }

    #[test]
    fn plain_string_unions_prefer_the_parsed_scalar() {
        for parameter in [
            serde_json::json!({"type": ["string", "integer"]}),
            serde_json::json!({"type": ["integer", "string"]}),
            serde_json::json!({"anyOf": [{"type": "string"}, {"type": "integer"}]}),
            serde_json::json!({"oneOf": [{"type": "integer"}, {"type": "string"}]}),
            serde_json::json!({"anyOf": [{"type": "string", "enum": ["auto"]}, {"type": "integer"}]}),
        ] {
            assert_eq!(
                parse_schema_value(parameter.clone(), "42"),
                serde_json::json!(42),
                "{parameter}"
            );
            assert_eq!(
                parse_schema_value(parameter.clone(), "-4200"),
                serde_json::json!(-4200),
                "{parameter}"
            );
            assert_eq!(
                parse_schema_value(parameter.clone(), "auto"),
                serde_json::json!("auto"),
                "{parameter}"
            );
            assert_eq!(parse_schema_value(parameter, "4a"), serde_json::json!("4a"));
        }
        for parameter in [
            serde_json::json!({"type": ["string", "number"]}),
            serde_json::json!({"anyOf": [{"type": "number"}, {"type": "string"}]}),
        ] {
            assert_eq!(
                parse_schema_value(parameter.clone(), "21.5"),
                serde_json::json!(21.5),
                "{parameter}"
            );
            assert_eq!(
                parse_schema_value(parameter, "-4.5"),
                serde_json::json!(-4.5)
            );
        }
        for parameter in [
            serde_json::json!({"type": ["string", "boolean"]}),
            serde_json::json!({"anyOf": [{"type": "boolean"}, {"type": "string"}]}),
        ] {
            assert_eq!(
                parse_schema_value(parameter.clone(), "false"),
                serde_json::json!(false),
                "{parameter}"
            );
            assert_eq!(
                parse_schema_value(parameter, "maybe"),
                serde_json::json!("maybe")
            );
        }
        // A bare string type still takes the text verbatim.
        assert_eq!(
            parse_schema_value(serde_json::json!({"type": "string"}), "42"),
            serde_json::json!("42")
        );
    }

    #[test]
    fn untyped_schemas_take_the_parsed_scalar() {
        let defs = serde_json::json!({"Any": {}, "Wrapped": {"$ref": "#/$defs/Any"}});
        for parameter in [
            serde_json::json!({}),
            serde_json::json!({"description": "no type"}),
            serde_json::json!({"minimum": 0}),
            serde_json::json!({"anyOf": [{}, {"type": "null"}]}),
            serde_json::json!({"allOf": [{}]}),
            serde_json::json!({"$ref": "#/$defs/Wrapped"}),
        ] {
            let tools = vec![ToolDefinition {
                name: "probe".into(),
                parameters: Some(serde_json::json!({
                    "type": "object", "properties": {"value": parameter}, "$defs": defs
                })),
                strict: None,
            }];
            for (raw, expected) in [
                ("37", serde_json::json!(37)),
                ("-0.75", serde_json::json!(-0.75)),
                ("false", serde_json::json!(false)),
                ("hello", serde_json::json!("hello")),
            ] {
                let wire = format!(
                    "<tool_call>probe<arg_key>value</arg_key><arg_value>{raw}</arg_value></tool_call>"
                );
                let (calls, _) =
                    try_tool_call_parse_glm47(&wire, &get_test_config(), Some(&tools)).unwrap();
                let arguments: Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
                assert_eq!(
                    arguments["value"], expected,
                    "schema={parameter}, raw={raw}"
                );
            }
        }
    }

    #[test]
    fn not_and_conditional_schemas_hint_types() {
        assert_eq!(
            parse_schema_value(serde_json::json!({"not": {"type": "string"}}), "-2500"),
            serde_json::json!(-2500)
        );
        assert_eq!(
            parse_schema_value(serde_json::json!({"not": {"type": "integer"}}), "2500"),
            serde_json::json!("2500")
        );
        let conditional = serde_json::json!({
            "if": {"type": "integer"}, "then": {"minimum": -10000}, "else": {"type": "string"}
        });
        assert_eq!(
            parse_schema_value(conditional.clone(), "-2500"),
            serde_json::json!(-2500)
        );
        assert_eq!(
            parse_schema_value(conditional, "later"),
            serde_json::json!("later")
        );
    }

    #[test]
    fn root_combinators_resolve_parameter_schemas() {
        let wire = |name: &str, key: &str, raw: &str| {
            format!(
                "<tool_call>{name}<arg_key>{key}</arg_key><arg_value>{raw}</arg_value></tool_call>"
            )
        };
        let parse = |parameters: Value, key: &str, raw: &str| -> Value {
            let tools = vec![ToolDefinition {
                name: "probe".into(),
                parameters: Some(parameters),
                strict: None,
            }];
            let (calls, _) = try_tool_call_parse_glm47(
                &wire("probe", key, raw),
                &get_test_config(),
                Some(&tools),
            )
            .unwrap();
            let arguments: Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
            arguments[key].clone()
        };
        let ticket = serde_json::json!({"type": "object", "properties": {"ticket_id": {"type": "integer"}},
            "required": ["ticket_id"], "additionalProperties": false});
        let reference = serde_json::json!({"type": "object", "properties": {"reference_code": {"type": "string"}},
            "required": ["reference_code"], "additionalProperties": false});
        for root in [
            serde_json::json!({"anyOf": [ticket, reference]}),
            serde_json::json!({"oneOf": [reference, ticket]}),
            serde_json::json!({"allOf": [ticket]}),
            serde_json::json!({"type": "object", "anyOf": [ticket, reference]}),
            serde_json::json!({"$ref": "#/$defs/Args", "$defs": {"Args": ticket}}),
            serde_json::json!({"if": {"required": ["ticket_id"]}, "then": ticket, "else": reference}),
        ] {
            assert_eq!(
                parse(root.clone(), "ticket_id", "48213"),
                serde_json::json!(48213),
                "{root}"
            );
            assert_eq!(
                parse(root, "reference_code", "TCK-1"),
                serde_json::json!("TCK-1")
            );
        }
        // The same key typed differently per branch is a union of the branches.
        let either = serde_json::json!({"anyOf": [
            {"type": "object", "properties": {"id": {"type": "integer"}}},
            {"type": "object", "properties": {"id": {"type": "string", "enum": ["latest"]}}}
        ]});
        assert_eq!(parse(either.clone(), "id", "7"), serde_json::json!(7));
        assert_eq!(parse(either, "id", "latest"), serde_json::json!("latest"));
    }

    #[test]
    fn quoted_scalars_coerce_when_the_schema_admits_no_string() {
        for parameter in [
            serde_json::json!({"type": "number"}),
            serde_json::json!({"anyOf": [{"type": "number"}, {"type": "null"}]}),
            serde_json::json!({"type": ["number", "null"]}),
        ] {
            assert_eq!(
                parse_schema_value(parameter.clone(), "\"50\""),
                serde_json::json!(50),
                "{parameter}"
            );
        }
        assert_eq!(
            parse_schema_value(serde_json::json!({"type": ["boolean", "null"]}), "\"true\""),
            serde_json::json!(true)
        );
        // A schema that admits a string keeps the quoted text verbatim.
        assert_eq!(
            parse_schema_value(serde_json::json!({"type": "string"}), "\"50\""),
            serde_json::json!("\"50\"")
        );
        assert_eq!(
            parse_schema_value(
                serde_json::json!({"anyOf": [{"type": "string"}, {"type": "number"}]}),
                "\"50\""
            ),
            serde_json::json!("\"50\"")
        );
        // Quoted text that is not the admitted scalar stays as it was.
        assert_eq!(
            parse_schema_value(serde_json::json!({"type": "number"}), "\"fifty\""),
            serde_json::json!("fifty")
        );
    }

    #[test]
    fn omitted_required_nullable_keys_are_filled_with_null() {
        let tools = vec![ToolDefinition {
            name: "read".into(),
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "offset": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
                    "limit": {"type": ["integer", "null"]},
                    "mode": {"type": "string"}
                },
                "required": ["path", "offset", "limit", "mode"],
                "additionalProperties": false
            })),
            strict: None,
        }];
        let wire =
            "<tool_call>read<arg_key>path</arg_key><arg_value>src/b.ts</arg_value></tool_call>";
        let (calls, _) = try_tool_call_parse_glm47(wire, &get_test_config(), Some(&tools)).unwrap();
        let arguments: Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        // Nullable required keys are present as null; a required string stays absent.
        assert_eq!(
            arguments,
            serde_json::json!({"path": "src/b.ts", "offset": null, "limit": null})
        );
        let wire = "<tool_call>read<arg_key>path</arg_key><arg_value>a</arg_value><arg_key>limit</arg_key><arg_value>5</arg_value></tool_call>";
        let (calls, _) = try_tool_call_parse_glm47(wire, &get_test_config(), Some(&tools)).unwrap();
        let arguments: Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(
            arguments,
            serde_json::json!({"path": "a", "limit": 5, "offset": null})
        );
    }

    #[test]
    fn inline_unions_preserve_string_hints_within_node_budget() {
        let mut parameter = serde_json::json!({"type": "string"});
        for keyword in ["anyOf", "oneOf", "allOf"].into_iter().cycle().take(16) {
            parameter = serde_json::json!({keyword: [parameter]});
        }
        let tools = vec![ToolDefinition {
            name: "set_label".into(),
            parameters: Some(null_ref_parameters(parameter)),
            strict: None,
        }];
        let wire = "<tool_call>set_label<arg_key>label</arg_key><arg_value>{\"x\":1}</arg_value></tool_call>";
        let (calls, _) = try_tool_call_parse_glm47(wire, &get_test_config(), Some(&tools)).unwrap();
        let arguments: Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(arguments["label"], serde_json::json!("{\"x\":1}"));
    }

    #[test]
    fn type_hints_bound_reference_depth() {
        for (references, expected) in [(16, serde_json::json!(42)), (17, serde_json::json!("42"))] {
            let mut definitions = serde_json::Map::new();
            for index in 0..references {
                definitions.insert(
                    format!("step_{index}"),
                    if index + 1 == references {
                        serde_json::json!({"type": "integer"})
                    } else {
                        serde_json::json!({"$ref": format!("#/$defs/step_{}", index + 1)})
                    },
                );
            }
            let tools = vec![ToolDefinition {
                name: "set_label".into(),
                parameters: Some(serde_json::json!({
                    "type": "object", "properties": {"label": {"$ref": "#/$defs/step_0"}},
                    "$defs": definitions
                })),
                strict: None,
            }];
            let wire =
                "<tool_call>set_label<arg_key>label</arg_key><arg_value>42</arg_value></tool_call>";
            let (calls, _) =
                try_tool_call_parse_glm47(wire, &get_test_config(), Some(&tools)).unwrap();
            let arguments: Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
            assert_eq!(arguments["label"], expected, "{references} references");
        }
    }

    #[test]
    fn branching_reference_cycles_exhaust_a_shared_budget() {
        let reference = serde_json::json!({"$ref": "#/$defs/Cycle"});
        let schema = serde_json::json!({
            "$defs": {"Cycle": {"anyOf": vec![reference.clone(); 8]}},
            "properties": {"value": reference}
        });
        let mut remaining = 64;
        assert_eq!(
            schema_type_match(
                &schema,
                &schema["properties"]["value"],
                "string",
                "value",
                0,
                &mut remaining
            ),
            None
        );
        assert_eq!(remaining, 0);
    }

    #[test]
    fn referenced_types_intersect_siblings_and_bound_cycles() {
        let schema = serde_json::json!({
            "$defs": {
                "Scalar": {"type": ["string", "integer"]},
                "postal code": {"type": "integer"},
                "café+": {"type": "integer"},
                "a/b~c": {"type": "integer"},
                "Text": {"type": "string"},
                "Nullable": {"type": ["string", "null"]},
                "Numbers": {"type": ["integer", "number"]},
                "NumbersReverse": {"type": ["number", "integer"]},
                "NumberOrBoolean": {"anyOf": [{"type": "number"}, {"type": "boolean"}]},
                "Alias": {"$ref": "#/$defs/postal%20code"},
                "Scoped": {
                    "$id": "https://example.com/scoped",
                    "$defs": {"postal code": {"type": "string"}},
                    "$ref": "#/$defs/postal%20code"
                },
                "Dynamic": {"$dynamicRef": "#/$defs/postal%20code"},
                "Recursive": {"$recursiveRef": "#"},
                "Loop": {"$ref": "#/$defs/Loop"}
            },
            "properties": {
                "narrow": {"$ref": "#/$defs/Scalar", "type": "integer"},
                "encoded": {"$ref": "#/$defs/postal%20code"},
                "utf8": {"$ref": "#/$defs/caf%c3%a9+"},
                "invalid_utf8": {"$ref": "#/$defs/%FF"},
                "incomplete": {"$ref": "#/$defs/%2"},
                "invalid_hex": {"$ref": "#/$defs/%GG"},
                "escaped": {"$ref": "#/$defs/a%7E1b%7E0c"},
                "cycle": {"$ref": "#/$defs/Loop"},
                "typed_cycle": {"$ref": "#/$defs/Loop", "type": "integer"},
                "payload": {"$ref": "#/$defs/Text"},
                "nullable": {"$ref": "#/$defs/Nullable"},
                "literal_null": {"$ref": "#/$defs/Text"},
                "integer_first": {"$ref": "#/$defs/Numbers"},
                "number_first": {"$ref": "#/$defs/NumbersReverse"},
                "bool_union": {"$ref": "#/$defs/NumberOrBoolean"},
                "allof_narrow": {"$ref": "#/$defs/Scalar", "allOf": [{"type": "integer"}]},
                "chain": {"$ref": "#/$defs/Alias"},
                "scoped": {"$ref": "#/$defs/Scoped"},
                "dynamic": {"$ref": "#/$defs/Dynamic"},
                "recursive": {"$ref": "#/$defs/Recursive"}
            }
        });
        for (field, raw, expected) in [
            ("narrow", "42", serde_json::json!(42)),
            ("encoded", "42", serde_json::json!(42)),
            ("utf8", "42", serde_json::json!(42)),
            ("invalid_utf8", "42", serde_json::json!("42")),
            ("incomplete", "42", serde_json::json!("42")),
            ("invalid_hex", "42", serde_json::json!("42")),
            ("escaped", "42", serde_json::json!(42)),
            ("cycle", "42", serde_json::json!("42")),
            ("typed_cycle", "42", serde_json::json!(42)),
            ("payload", "{\"x\":1}", serde_json::json!("{\"x\":1}")),
            ("nullable", "null", serde_json::json!(null)),
            ("literal_null", "null", serde_json::json!("null")),
            ("integer_first", "3.5", serde_json::json!(3.5)),
            ("number_first", "42", serde_json::json!(42)),
            ("bool_union", "true", serde_json::json!(true)),
            ("allof_narrow", "42", serde_json::json!(42)),
            ("chain", "42", serde_json::json!(42)),
            ("scoped", "42", serde_json::json!("42")),
            ("dynamic", "42", serde_json::json!("42")),
            ("recursive", "42", serde_json::json!("42")),
        ] {
            let tools = vec![ToolDefinition {
                name: "capture_payload".into(),
                strict: None,
                parameters: Some(schema.clone()),
            }];
            let input = format!(
                "<tool_call>capture_payload<arg_key>{field}</arg_key><arg_value>{raw}</arg_value></tool_call>"
            );
            let (calls, _) =
                try_tool_call_parse_glm47(&input, &get_test_config(), Some(&tools)).unwrap();
            let args: Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
            assert_eq!(args[field], expected, "{field}");
        }
    }

    fn parse_null_with_parameters(parameters: Value) -> Value {
        let tools = vec![ToolDefinition {
            name: "set_label".into(),
            parameters: Some(parameters),
            strict: None,
        }];
        let wire =
            "<tool_call>set_label<arg_key>label</arg_key><arg_value>null</arg_value></tool_call>";
        let (calls, _) = try_tool_call_parse_glm47(wire, &get_test_config(), Some(&tools)).unwrap();
        assert_eq!(calls.len(), 1);
        let arguments: Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        arguments["label"].clone()
    }

    fn null_ref_parameters(parameter: Value) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {"label": parameter, "other": {"type": "string"}},
            "$defs": {
                "String": {"type": "string"},
                "Nullable": {"type": ["string", "null"]},
                "Alias": {"$ref": "#/$defs/Nullable"},
                "True": true,
                "False": false,
                "CycleA": {"$ref": "#/$defs/CycleB"},
                "CycleB": {"$ref": "#/$defs/CycleA"},
                "Slash/Tilde~": {"type": "string"},
                "café name": {"type": "string"},
                "Bad~2": {"type": ["string", "null"]},
                "NotSchema": null,
                "Scoped": {
                    "$id": "nested",
                    "$defs": {"Label": {"type": ["string", "null"]}}
                }
            },
            "definitions": {"String": {"type": "string"}}
        })
    }

    #[test]
    fn null_coercion_resolves_local_refs() {
        for (label, parameter, permits_null) in [
            (
                "string",
                serde_json::json!({"$ref": "#/$defs/String"}),
                false,
            ),
            (
                "nullable",
                serde_json::json!({"$ref": "#/$defs/Nullable"}),
                true,
            ),
            ("chain", serde_json::json!({"$ref": "#/$defs/Alias"}), true),
            (
                "definitions",
                serde_json::json!({"$ref": "#/definitions/String"}),
                false,
            ),
            (
                "property",
                serde_json::json!({"$ref": "#/properties/other"}),
                false,
            ),
            ("root", serde_json::json!({"$ref": "#"}), false),
            ("true", serde_json::json!({"$ref": "#/$defs/True"}), true),
            ("false", serde_json::json!({"$ref": "#/$defs/False"}), false),
            (
                "escaped",
                serde_json::json!({"$ref": "#/$defs/Slash~1Tilde~0"}),
                false,
            ),
            (
                "percent_encoded",
                serde_json::json!({"$ref": "#/$defs/caf%C3%A9%20name"}),
                false,
            ),
            (
                "encoded_escapes",
                serde_json::json!({"$ref": "#/$defs/Slash%7E1Tilde%7E0"}),
                false,
            ),
            (
                "sibling_type",
                serde_json::json!({"$ref": "#/$defs/Nullable", "type": "string"}),
                false,
            ),
            (
                "sibling_enum",
                serde_json::json!({"$ref": "#/$defs/Nullable", "enum": ["null"]}),
                false,
            ),
            (
                "sibling_const",
                serde_json::json!({"$ref": "#/$defs/Nullable", "const": "null"}),
                false,
            ),
            (
                "sibling_not",
                serde_json::json!({"$ref": "#/$defs/Nullable", "not": {"type": "null"}}),
                false,
            ),
            (
                "allof",
                serde_json::json!({"allOf": [{"$ref": "#/$defs/Nullable"}, {"$ref": "#/$defs/String"}]}),
                false,
            ),
            (
                "anyof",
                serde_json::json!({"anyOf": [{"$ref": "#/$defs/String"}, {"type": "null"}]}),
                true,
            ),
            (
                "oneof",
                serde_json::json!({"oneOf": [{"$ref": "#/$defs/String"}, {"$ref": "#/$defs/Nullable"}]}),
                true,
            ),
            (
                "not_string",
                serde_json::json!({"not": {"$ref": "#/$defs/String"}}),
                true,
            ),
            (
                "repeated_ref_anyof",
                serde_json::json!({"anyOf": [{"$ref": "#/$defs/Nullable"}, {"$ref": "#/$defs/Nullable"}]}),
                true,
            ),
            (
                "repeated_ref_oneof",
                serde_json::json!({"oneOf": [{"$ref": "#/$defs/Nullable"}, {"$ref": "#/$defs/Nullable"}]}),
                false,
            ),
        ] {
            let expected = if permits_null {
                Value::Null
            } else {
                serde_json::json!("null")
            };
            assert_eq!(
                parse_null_with_parameters(null_ref_parameters(parameter)),
                expected,
                "{label}"
            );
        }
        let mut root_with_id = null_ref_parameters(serde_json::json!({"$ref": "#/$defs/Nullable"}));
        root_with_id["$id"] = serde_json::json!("https://example.com/parameters");
        assert_eq!(parse_null_with_parameters(root_with_id), Value::Null);
    }

    #[test]
    fn null_coercion_schema_map_names_are_not_scopes() {
        for keyword in ["$id", "$dynamicRef", "$recursiveRef"] {
            for definition in [
                serde_json::json!({"type": "string"}),
                serde_json::json!(true),
            ] {
                let mut parameters =
                    null_ref_parameters(serde_json::json!({"$ref": "#/$defs/Nullable"}));
                parameters["$defs"][keyword] = definition;
                assert_eq!(
                    parse_null_with_parameters(parameters),
                    Value::Null,
                    "{keyword}"
                );
            }
        }
    }

    #[test]
    fn null_coercion_preserves_unknown_refs() {
        for reference in [
            "#/$defs/Missing",
            "https://example.com/schema",
            "#anchor",
            "#/$defs/Bad~2",
            "#/$defs/Nullable%",
            "#/$defs/Nullable%GG",
            "#/$defs/%FF",
            "#/$defs/NotSchema",
            "#/$defs/CycleA",
            "#/properties/label",
            "#/$defs/Scoped/$defs/Label",
        ] {
            for parameter in [
                serde_json::json!({"$ref": reference}),
                serde_json::json!({"not": {"$ref": reference}}),
                serde_json::json!({"oneOf": [{"type": "null"}, {"$ref": reference}]}),
            ] {
                assert_eq!(
                    parse_null_with_parameters(null_ref_parameters(parameter.clone())),
                    serde_json::json!("null"),
                    "{parameter}"
                );
            }
        }
        for parameter in [
            serde_json::json!({"$ref": 42}),
            serde_json::json!({"$id": "nested", "$ref": "#/$defs/Nullable"}),
            serde_json::json!({"$dynamicRef": "#/$defs/Nullable"}),
            serde_json::json!({"$recursiveRef": "#"}),
        ] {
            assert_eq!(
                parse_null_with_parameters(null_ref_parameters(parameter.clone())),
                serde_json::json!("null"),
                "{parameter}"
            );
        }
    }

    #[test]
    fn null_coercion_combines_unknown_ref_branches() {
        for (parameter, permits_null) in [
            (
                serde_json::json!({"anyOf": [{"type": "null"}, {"$ref": "#/$defs/Missing"}]}),
                true,
            ),
            (
                serde_json::json!({"allOf": [{"type": "null"}, {"$ref": "#/$defs/Missing"}]}),
                false,
            ),
            (
                serde_json::json!({"oneOf": [{"type": "null"}, {"$ref": "#/$defs/Missing"}]}),
                false,
            ),
            (
                serde_json::json!({"not": {"allOf": [{"type": "string"}, {"$ref": "#/$defs/Missing"}]}}),
                true,
            ),
            (
                serde_json::json!({"not": {"anyOf": [{"type": "string"}, {"$ref": "#/$defs/Missing"}]}}),
                false,
            ),
            (
                serde_json::json!({"not": {"oneOf": [{"type": "null"}, {"type": "null"}, {"$ref": "#/$defs/Missing"}]}}),
                true,
            ),
        ] {
            let expected = if permits_null {
                Value::Null
            } else {
                serde_json::json!("null")
            };
            assert_eq!(
                parse_null_with_parameters(null_ref_parameters(parameter.clone())),
                expected,
                "{parameter}"
            );
        }
    }

    #[test]
    fn null_coercion_bounds_reference_expansion() {
        for (references, expected) in [(16, Value::Null), (17, serde_json::json!("null"))] {
            let mut definitions = serde_json::Map::new();
            for index in 0..references {
                definitions.insert(
                    format!("step_{index}"),
                    if index + 1 == references {
                        serde_json::json!({"type": "null"})
                    } else {
                        serde_json::json!({"$ref": format!("#/$defs/step_{}", index + 1)})
                    },
                );
            }
            let parameters = serde_json::json!({
                "type": "object", "properties": {"label": {"$ref": "#/$defs/step_0"}},
                "$defs": definitions
            });
            assert_eq!(
                parse_null_with_parameters(parameters),
                expected,
                "{references} references"
            );
        }
        let parameter =
            serde_json::json!({"allOf": vec![serde_json::json!({"type": "null"}); 4096]});
        assert_eq!(
            parse_null_with_parameters(null_ref_parameters(parameter)),
            serde_json::json!("null")
        );
    }

    #[test]
    fn nullable_string_union_preserves_json_null() {
        let tools = vec![ToolDefinition {
            name: "set_labels".to_string(),
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": {
                    "label": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                    "note": {"type": ["string", "null"]},
                    "literal": {"type": "string"}
                }
            })),
            strict: None,
        }];
        let message = concat!(
            "<tool_call>set_labels",
            "<arg_key>label</arg_key><arg_value>null</arg_value>",
            "<arg_key>note</arg_key><arg_value>null</arg_value>",
            "<arg_key>literal</arg_key><arg_value>null</arg_value>",
            "</tool_call>"
        );
        let (calls, _) =
            try_tool_call_parse_glm47(message, &get_test_config(), Some(&tools)).unwrap();
        assert_eq!(calls.len(), 1);
        let args: Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(
            args,
            serde_json::json!({"label": null, "note": null, "literal": "null"})
        );
    }

    #[test]
    fn test_parse_newline_between_arg_tags() {
        let config = get_test_config();
        let message = "<tool_call>get_weather\n<arg_key>city</arg_key>\n<arg_value>Paris</arg_value>\n</tool_call>\n<tool_call>get_time\n<arg_key>city</arg_key>\n<arg_value>London</arg_value>\n</tool_call>";

        let (calls, _) = try_tool_call_parse_glm47(message, &config, None).unwrap();

        assert_eq!(calls.len(), 2);
        for (call, (name, city)) in calls
            .iter()
            .zip([("get_weather", "Paris"), ("get_time", "London")])
        {
            assert_eq!(call.function.name, name);
            let args: HashMap<String, Value> =
                serde_json::from_str(&call.function.arguments).unwrap();
            assert_eq!(args.get("city").unwrap().as_str().unwrap(), city);
        }
    }

    #[test] // helper
    fn test_detect_tool_call_start() {
        let config = get_test_config();

        // Complete start token
        assert!(detect_tool_call_start_glm47(
            "<tool_call>get_weather",
            &config
        ));

        // Partial start token (streaming)
        assert!(detect_tool_call_start_glm47("Some text <tool", &config));
        assert!(detect_tool_call_start_glm47("Some text <tool_c", &config));

        // No tool call
        assert!(!detect_tool_call_start_glm47("Just normal text", &config));
    }

    // DEPRECATED(parser-fixture-duplicate): Duplicate of YAML fixture coverage: TOOLCALLING.batch.1 in tests/parity/toolcalling/fixtures/glm47/TOOLCALLING.batch.yaml.
    #[test] // TOOLCALLING.batch.1
    fn test_parse_simple_tool_call() {
        let config = get_test_config();
        let message = "<tool_call>get_weather<arg_key>location</arg_key><arg_value>San Francisco</arg_value></tool_call>";

        let (calls, normal_text) = try_tool_call_parse_glm47(message, &config, None).unwrap();

        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");

        let args: HashMap<String, Value> =
            serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(
            args.get("location").unwrap().as_str().unwrap(),
            "San Francisco"
        );
        assert_eq!(normal_text, Some("".to_string()));
    }

    // DEPRECATED(parser-fixture-duplicate): Duplicate of YAML fixture coverage: TOOLCALLING.batch.1, TOOLCALLING.batch.7.d in tests/parity/toolcalling/fixtures/glm47/TOOLCALLING.batch.7.yaml, tests/parity/toolcalling/fixtures/glm47/TOOLCALLING.batch.yaml.
    #[test] // TOOLCALLING.batch.1, TOOLCALLING.batch.7
    fn test_parse_tool_call_with_multiple_args() {
        let config = get_test_config();
        let message = "<tool_call>book_flight<arg_key>from</arg_key><arg_value>NYC</arg_value><arg_key>to</arg_key><arg_value>LAX</arg_value><arg_key>date</arg_key><arg_value>2026-03-15</arg_value></tool_call>";

        let (calls, _) = try_tool_call_parse_glm47(message, &config, None).unwrap();

        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "book_flight");

        let args: HashMap<String, Value> =
            serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args.get("from").unwrap().as_str().unwrap(), "NYC");
        assert_eq!(args.get("to").unwrap().as_str().unwrap(), "LAX");
        assert_eq!(args.get("date").unwrap().as_str().unwrap(), "2026-03-15");
    }

    // DEPRECATED(parser-fixture-duplicate): Duplicate of YAML fixture coverage: TOOLCALLING.batch.7.d in tests/parity/toolcalling/fixtures/glm47/TOOLCALLING.batch.7.yaml.
    #[test] // TOOLCALLING.batch.7
    fn test_parse_tool_call_with_json_value() {
        let config = get_test_config();
        let message = r#"<tool_call>search<arg_key>filters</arg_key><arg_value>{"category": "books", "price_max": 50}</arg_value></tool_call>"#;

        let (calls, _) = try_tool_call_parse_glm47(message, &config, None).unwrap();

        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "search");

        let args: HashMap<String, Value> =
            serde_json::from_str(&calls[0].function.arguments).unwrap();
        let filters = args.get("filters").unwrap();
        assert!(filters.is_object());
    }

    // DEPRECATED(parser-fixture-duplicate): Duplicate of YAML fixture coverage: TOOLCALLING.batch.2.b in tests/parity/toolcalling/fixtures/glm47/TOOLCALLING.batch.2.yaml.
    #[test] // TOOLCALLING.batch.2
    fn test_parse_multiple_tool_calls() {
        let config = get_test_config();
        let message = "<tool_call>get_weather<arg_key>location</arg_key><arg_value>NYC</arg_value></tool_call><tool_call>get_time<arg_key>timezone</arg_key><arg_value>EST</arg_value></tool_call>";

        let (calls, _) = try_tool_call_parse_glm47(message, &config, None).unwrap();

        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[1].function.name, "get_time");
    }

    // DEPRECATED(parser-fixture-duplicate): Duplicate of YAML fixture coverage: TOOLCALLING.batch.8.a in tests/parity/toolcalling/fixtures/glm47/TOOLCALLING.batch.8.yaml.
    #[test] // TOOLCALLING.batch.8
    fn test_parse_with_normal_text() {
        let config = get_test_config();
        let message = "I'll check the weather for you. <tool_call>get_weather<arg_key>location</arg_key><arg_value>Paris</arg_value></tool_call>";

        let (calls, normal_text) = try_tool_call_parse_glm47(message, &config, None).unwrap();

        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(
            normal_text,
            Some("I'll check the weather for you. ".to_string())
        );
    }

    // DEPRECATED(parser-fixture-duplicate): Duplicate of YAML fixture coverage: TOOLCALLING.batch.6.a in tests/parity/toolcalling/fixtures/glm47/TOOLCALLING.batch.6.yaml.
    #[test] // TOOLCALLING.batch.6
    fn test_parse_tool_call_no_args() {
        let config = get_test_config();
        let message = "<tool_call>get_current_time</tool_call>";

        let (calls, _) = try_tool_call_parse_glm47(message, &config, None).unwrap();

        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_current_time");

        let args: HashMap<String, Value> =
            serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert!(args.is_empty());
    }

    #[test] // helper
    fn test_find_tool_call_end_position() {
        let config = get_test_config();
        let chunk =
            "<tool_call>func<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>more text";

        let end_pos = find_tool_call_end_position_glm47(chunk, &config);
        assert_eq!(
            &chunk[..end_pos],
            "<tool_call>func<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>"
        );
    }

    #[test] // helper — parallel calls: end position must advance past ALL blocks
    fn test_find_tool_call_end_position_parallel() {
        let config = get_test_config();
        let chunk = "<tool_call>get_weather<arg_key>location</arg_key><arg_value>SF</arg_value></tool_call><tool_call>get_weather<arg_key>location</arg_key><arg_value>NYC</arg_value></tool_call>trailing";

        let end_pos = find_tool_call_end_position_glm47(chunk, &config);
        assert_eq!(
            &chunk[..end_pos],
            "<tool_call>get_weather<arg_key>location</arg_key><arg_value>SF</arg_value></tool_call><tool_call>get_weather<arg_key>location</arg_key><arg_value>NYC</arg_value></tool_call>",
            "Must advance past ALL consecutive tool call blocks"
        );
    }

    #[test] // helper — parallel calls with whitespace between blocks
    fn test_find_tool_call_end_position_parallel_with_whitespace() {
        let config = get_test_config();
        let chunk = "<tool_call>a<arg_key>k</arg_key><arg_value>1</arg_value></tool_call>\n<tool_call>b<arg_key>k</arg_key><arg_value>2</arg_value></tool_call>";

        let end_pos = find_tool_call_end_position_glm47(chunk, &config);
        assert_eq!(
            end_pos,
            chunk.len(),
            "Must handle whitespace/newlines between consecutive blocks"
        );
    }

    #[test] // helper — parallel calls: second block incomplete (streaming)
    fn test_find_tool_call_end_position_parallel_second_incomplete() {
        let config = get_test_config();
        let chunk = "<tool_call>a<arg_key>k</arg_key><arg_value>1</arg_value></tool_call><tool_call>b<arg_key>k</arg_key><arg_value>2</arg_value>";

        let end_pos = find_tool_call_end_position_glm47(chunk, &config);
        assert_eq!(
            &chunk[..end_pos],
            "<tool_call>a<arg_key>k</arg_key><arg_value>1</arg_value></tool_call>",
            "Must stop at first complete block when second is incomplete"
        );
    }

    // DEPRECATED(parser-fixture-duplicate): Duplicate of YAML fixture coverage: TOOLCALLING.batch.2.c, TOOLCALLING.batch.8.a in tests/parity/toolcalling/fixtures/glm47/TOOLCALLING.batch.2.yaml, tests/parity/toolcalling/fixtures/glm47/TOOLCALLING.batch.8.yaml.
    #[test] // TOOLCALLING.batch.2 + TOOLCALLING.batch.8 — bug report repro: text + parallel calls
    fn test_parse_text_then_parallel_calls() {
        let config = get_test_config();
        let message = "I'll check the weather for both cities at the same time!<tool_call>get_weather<arg_key>location</arg_key><arg_value>San Francisco</arg_value></tool_call><tool_call>get_weather<arg_key>location</arg_key><arg_value>New York</arg_value></tool_call>";

        let (calls, normal_text) = try_tool_call_parse_glm47(message, &config, None).unwrap();

        assert_eq!(calls.len(), 2, "Both parallel calls must be extracted");
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[1].function.name, "get_weather");

        let args0: HashMap<String, Value> =
            serde_json::from_str(&calls[0].function.arguments).unwrap();
        let args1: HashMap<String, Value> =
            serde_json::from_str(&calls[1].function.arguments).unwrap();
        assert_eq!(
            args0.get("location").unwrap().as_str().unwrap(),
            "San Francisco"
        );
        assert_eq!(args1.get("location").unwrap().as_str().unwrap(), "New York");

        let text = normal_text.unwrap();
        assert_eq!(
            text,
            "I'll check the weather for both cities at the same time!"
        );
    }

    #[test]
    fn test_parse_two_bare_calls_without_start_markers_recovers_independently() {
        let config = get_test_config();
        let message = "I will check both. get_weather<arg_key>location</arg_key><arg_value>NYC</arg_value></tool_call>get_time<arg_key>timezone</arg_key><arg_value>EST</arg_value></tool_call>";

        let (calls, normal_text) = try_tool_call_parse_glm47(message, &config, None).unwrap();

        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[1].function.name, "get_time");
        let args0: HashMap<String, Value> =
            serde_json::from_str(&calls[0].function.arguments).unwrap();
        let args1: HashMap<String, Value> =
            serde_json::from_str(&calls[1].function.arguments).unwrap();
        assert_eq!(args0["location"], "NYC");
        assert_eq!(args1["timezone"], "EST");
        assert_eq!(normal_text.unwrap(), "I will check both. ");
    }

    // DEPRECATED(parser-fixture-duplicate): Duplicate of YAML fixture coverage: TOOLCALLING.batch.7.b in tests/parity/toolcalling/fixtures/glm47/TOOLCALLING.batch.7.yaml.
    #[test] // TOOLCALLING.batch.7, TOOLCALLING.fmt.2
    fn test_parse_multiline_arg_value() {
        let config = get_test_config();
        let message = "<tool_call>write_file<arg_key>path</arg_key><arg_value>/tmp/hello.py</arg_value><arg_key>content</arg_key><arg_value>#!/usr/bin/env python3\nprint(\"Hello, World!\")\n</arg_value></tool_call>";

        let (calls, _) = try_tool_call_parse_glm47(message, &config, None).unwrap();

        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "write_file");

        let args: HashMap<String, Value> =
            serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args.get("path").unwrap().as_str().unwrap(), "/tmp/hello.py");
        assert!(
            args.contains_key("content"),
            "content argument must be parsed even when it contains newlines"
        );
        let content = args.get("content").unwrap().as_str().unwrap();
        assert!(content.contains("print(\"Hello, World!\")"));
    }

    // DEPRECATED(parser-fixture-duplicate): Duplicate of YAML fixture coverage: TOOLCALLING.batch.4.d in tests/parity/toolcalling/fixtures/glm47/TOOLCALLING.batch.4.yaml.
    #[test] // TOOLCALLING.batch.4
    fn test_malformed_tool_call() {
        let config = get_test_config();

        // Missing end tag
        let message = "<tool_call>get_weather";
        let result = try_tool_call_parse_glm47(message, &config, None);
        assert!(result.is_ok()); // Should handle gracefully, no calls extracted

        let (calls, _) = result.unwrap();
        assert_eq!(calls.len(), 0);
    }

    // Recovery for missing outer </tool_call> (max_tokens / EOS truncation):
    // when the inner arg pairs are well-formed, treat EOF as the end token
    // and extract the call. The arg_key opener gates recovery so plain text
    // that happens to start with `<tool_call>` is still preserved verbatim.
    // DEPRECATED(parser-fixture-duplicate): Duplicate of YAML fixture coverage: TOOLCALLING.batch.5.a in tests/parity/toolcalling/fixtures/glm47/TOOLCALLING.batch.5.yaml.
    #[test] // TOOLCALLING.batch.5
    fn test_parse_no_end_tag_complete_args_recovers() {
        let config = Glm47ParserConfig {
            allow_eof_recovery: true,
            ..get_test_config()
        };
        // Args complete, only outer </tool_call> missing.
        let message = "<tool_call>get_weather<arg_key>location</arg_key><arg_value>NYC</arg_value>";

        let (calls, _) = try_tool_call_parse_glm47(message, &config, None).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["location"], "NYC");
    }

    // DEPRECATED(parser-fixture-duplicate): Duplicate of YAML fixture coverage: TOOLCALLING.batch.2.b, TOOLCALLING.batch.5.a in tests/parity/toolcalling/fixtures/glm47/TOOLCALLING.batch.2.yaml, tests/parity/toolcalling/fixtures/glm47/TOOLCALLING.batch.5.yaml.
    #[test] // TOOLCALLING.batch.5
    fn test_parse_no_end_tag_multiple_calls_recovers() {
        let config = Glm47ParserConfig {
            allow_eof_recovery: true,
            ..get_test_config()
        };
        // Two complete inner calls, missing only the trailing </tool_call> on the second.
        let message = "<tool_call>get_weather<arg_key>city</arg_key><arg_value>NYC</arg_value></tool_call><tool_call>get_time<arg_key>tz</arg_key><arg_value>EST</arg_value>";

        let (calls, _) = try_tool_call_parse_glm47(message, &config, None).unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[1].function.name, "get_time");
    }

    // Fixture coverage: glm47/TOOLCALLING.batch.yaml (TOOLCALLING.batch.13).
    #[test]
    fn test_unknown_function_block_returned_as_call_no_tag_leak() {
        let config = get_test_config();
        let tools = vec![ToolDefinition {
            name: "get_weather".to_string(),
            parameters: None,
            strict: None,
        }];

        let message = "Here is the result: <tool_call>unknown_func<arg_key>x</arg_key><arg_value>1</arg_value></tool_call> done";
        let (calls, normal_text) =
            try_tool_call_parse_glm47(message, &config, Some(&tools)).unwrap();

        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "unknown_func");
        let args: Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args, serde_json::json!({"x": "1"}));
        let text = normal_text.unwrap();
        assert!(
            !text.contains("<tool_call>") && !text.contains("<arg_key>"),
            "Wire-format tags must not leak into normal_text, got: {text}"
        );
        assert!(
            text.contains("Here is the result:"),
            "Leading prose must survive, got: {text}"
        );
    }

    #[tokio::test]
    async fn test_public_dispatch_preserves_mixed_unknown_and_known_calls() {
        let tools = vec![ToolDefinition {
            name: "known_func".to_string(),
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": {"x": {"type": "integer"}}
            })),
            strict: None,
        }];
        let message = "<tool_call>unknown_func<arg_key>x</arg_key><arg_value>1</arg_value></tool_call><tool_call>known_func<arg_key>x</arg_key><arg_value>2</arg_value></tool_call>";
        let (calls, normal_text) =
            crate::tool_calling::parsers::detect_and_parse_tool_call_with_recovery(
                message,
                Some("glm47"),
                Some(&tools),
            )
            .await
            .unwrap();

        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].function.name, "unknown_func");
        assert_eq!(calls[1].function.name, "known_func");
        let unknown_args: Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        let known_args: Value = serde_json::from_str(&calls[1].function.arguments).unwrap();
        assert_eq!(unknown_args, serde_json::json!({"x": "1"}));
        assert_eq!(known_args, serde_json::json!({"x": 2}));
        assert!(normal_text.unwrap_or_default().trim().is_empty());
    }

    #[test] // helper
    fn test_xml_entities_kept_verbatim() {
        let config = get_test_config();
        let message = r#"<tool_call>write_file<arg_key>content</arg_key><arg_value>x &lt; y &amp;&amp; y &gt; z</arg_value></tool_call>"#;

        let (calls, _) = try_tool_call_parse_glm47(message, &config, None).unwrap();

        assert_eq!(calls.len(), 1);
        let args: HashMap<String, Value> =
            serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(
            args.get("content").unwrap().as_str().unwrap(),
            "x &lt; y &amp;&amp; y &gt; z"
        );
    }

    #[test] // helper
    fn test_type_coercion_with_schema() {
        let config = get_test_config();
        let tools = vec![ToolDefinition {
            name: "set_temperature".to_string(),
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": {
                    "degrees": {"type": "number"},
                    "enabled": {"type": "boolean"},
                    "count": {"type": "integer"},
                    "huge_integer": {"type": "integer"},
                    "large_count": {"type": "number"},
                    "huge_count": {"type": "number"},
                    "label": {"type": "string"}
                }
            })),
            strict: None,
        }];

        let message = "<tool_call>set_temperature<arg_key>degrees</arg_key><arg_value>72.5</arg_value><arg_key>enabled</arg_key><arg_value>true</arg_value><arg_key>count</arg_key><arg_value>3</arg_value><arg_key>huge_integer</arg_key><arg_value>9223372036854775808</arg_value><arg_key>large_count</arg_key><arg_value>9007199254740993</arg_value><arg_key>huge_count</arg_key><arg_value>100000000000000000000</arg_value><arg_key>label</arg_key><arg_value>warm</arg_value></tool_call>";

        let (calls, _) = try_tool_call_parse_glm47(message, &config, Some(&tools)).unwrap();
        assert_eq!(calls.len(), 1);

        let args: HashMap<String, Value> =
            serde_json::from_str(&calls[0].function.arguments).unwrap();

        // number coercion
        assert_eq!(args.get("degrees").unwrap().as_f64().unwrap(), 72.5);
        // boolean coercion
        assert!(args.get("enabled").unwrap().as_bool().unwrap());
        // integer coercion
        assert_eq!(args.get("count").unwrap().as_i64().unwrap(), 3);
        // integer-like numbers should not be rounded through f64
        assert_eq!(
            args.get("large_count").unwrap().as_i64().unwrap(),
            9007199254740993
        );
        let raw_args: HashMap<String, Box<serde_json::value::RawValue>> =
            serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(raw_args["huge_integer"].get(), "9223372036854775808");
        assert_eq!(raw_args["huge_count"].get(), "100000000000000000000");
        // string stays string
        assert_eq!(args.get("label").unwrap().as_str().unwrap(), "warm");
    }

    #[test]
    fn test_arguments_serialized_in_source_order() {
        let config = get_test_config();
        let message = concat!(
            "<tool_call>create_ticket",
            "<arg_key>title</arg_key><arg_value>Rotate keys</arg_value>",
            "<arg_key>description</arg_key><arg_value>Keys are 90 days old</arg_value>",
            "<arg_key>priority</arg_key><arg_value>medium</arg_value>",
            "<arg_key>labels</arg_key><arg_value>[\"security\", \"ops\"]</arg_value>",
            "<arg_key>title</arg_key><arg_value>Rotate S3 keys</arg_value>",
            "</tool_call>"
        );

        let (calls, _) = try_tool_call_parse_glm47(message, &config, None).unwrap();
        assert_eq!(calls.len(), 1);
        let keys: Vec<&str> = regex::Regex::new(r#""([a-z_]+)":"#)
            .unwrap()
            .captures_iter(&calls[0].function.arguments)
            .map(|c| c.get(1).unwrap().as_str())
            .collect();
        assert_eq!(keys, ["title", "description", "priority", "labels"]);
        let args: HashMap<String, Value> =
            serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["title"], Value::String("Rotate S3 keys".to_string()));
        assert_eq!(args["labels"], serde_json::json!(["security", "ops"]));
    }

    #[test]
    fn test_many_arguments_preserve_order_and_replace_duplicates() {
        const COUNT: usize = 4096;
        let replacements = [COUNT - 1, COUNT / 2, 0];
        let mut message = String::from("<tool_call>bulk_update");
        for i in (0..COUNT).rev() {
            message.push_str(&format!(
                "<arg_key>arg_{i}</arg_key><arg_value>initial</arg_value>"
            ));
        }
        for i in replacements {
            message.push_str(&format!(
                "<arg_key>arg_{i}</arg_key><arg_value>updated</arg_value>"
            ));
        }
        message.push_str("</tool_call>");

        let (calls, _) = try_tool_call_parse_glm47(&message, &get_test_config(), None).unwrap();
        assert_eq!(calls.len(), 1);
        let expected = (0..COUNT)
            .rev()
            .map(|i| {
                let value = if replacements.contains(&i) {
                    "updated"
                } else {
                    "initial"
                };
                format!(r#""arg_{i}":"{value}""#)
            })
            .collect::<Vec<_>>()
            .join(",");
        assert_eq!(calls[0].function.arguments, format!("{{{expected}}}"));
    }

    #[test]
    fn test_string_schema_keeps_json_looking_values_verbatim() {
        for param_schema in [
            serde_json::json!({"type": "string"}),
            serde_json::json!({"type": ["string"]}),
            serde_json::json!({"type": ["string", "null"]}),
            serde_json::json!({"type": ["null", "string"]}),
            serde_json::json!({"type": ["object", "array", "string"]}),
            serde_json::json!({"anyOf": [{"type": "string"}, {"type": "null"}]}),
            serde_json::json!({"oneOf": [{"type": "null"}, {"type": "string"}]}),
            serde_json::json!({"allOf": [{"type": "string"}]}),
            serde_json::json!({"allOf": [{"minLength": 1}, {"type": "string"}]}),
            serde_json::json!({"allOf": [
                {"anyOf": [
                    {"type": "null"},
                    {"oneOf": [{"type": "array"}, {"type": "string"}]}
                ]},
                {"minLength": 1}
            ]}),
            serde_json::json!({"oneOf": [
                {"type": "null"}, {"allOf": [{"type": "string"}, {"minLength": 1}]}
            ]}),
            serde_json::json!({"anyOf": [
                {"type": "object"},
                {"oneOf": [{"type": "array"}, {"type": ["null", "string"]}]}
            ]}),
        ] {
            let config = get_test_config();
            let tools = vec![ToolDefinition {
                name: "save_note".to_string(),
                parameters: Some(serde_json::json!({
                    "type": "object",
                    "properties": {
                        "object_text": param_schema,
                        "array_text": param_schema,
                        "quoted_text": param_schema,
                        "payload": {"type": "object"},
                        "composed_payload": {"allOf": [
                            {"type": "object"},
                            {"properties": {"key": {"type": "string"}}}
                        ]},
                        "untyped": {}
                    }
                })),
                strict: None,
            }];

            let message = concat!(
                "<tool_call>save_note",
                "<arg_key>object_text</arg_key><arg_value>{\"key\": \"value\"}</arg_value>",
                "<arg_key>array_text</arg_key><arg_value>[1, 2, 3]</arg_value>",
                "<arg_key>quoted_text</arg_key><arg_value>\"quoted\"</arg_value>",
                "<arg_key>payload</arg_key><arg_value>{\"key\": \"value\"}</arg_value>",
                "<arg_key>composed_payload</arg_key><arg_value>{\"key\": \"value\"}</arg_value>",
                "<arg_key>untyped</arg_key><arg_value>[1, 2, 3]</arg_value>",
                "</tool_call>"
            );

            let (calls, _) = try_tool_call_parse_glm47(message, &config, Some(&tools)).unwrap();
            assert_eq!(calls.len(), 1);
            let args: HashMap<String, Value> =
                serde_json::from_str(&calls[0].function.arguments).unwrap();

            assert_eq!(
                args["object_text"],
                Value::String("{\"key\": \"value\"}".to_string())
            );
            assert_eq!(args["array_text"], Value::String("[1, 2, 3]".to_string()));
            assert_eq!(args["quoted_text"], Value::String("\"quoted\"".to_string()));
            assert_eq!(args["payload"], serde_json::json!({"key": "value"}));
            assert_eq!(
                args["composed_payload"],
                serde_json::json!({"key": "value"})
            );
            assert_eq!(args["untyped"], serde_json::json!([1, 2, 3]));
        }
    }

    #[test] // helper
    fn test_type_coercion_array_comma_separated() {
        let config = get_test_config();
        let tools = vec![ToolDefinition {
            name: "tag_item".to_string(),
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": {
                    "tags": {"type": "array"}
                }
            })),
            strict: None,
        }];

        // Model emits comma-separated values without JSON brackets
        let message = "<tool_call>tag_item<arg_key>tags</arg_key><arg_value>rust, python, go</arg_value></tool_call>";
        let (calls, _) = try_tool_call_parse_glm47(message, &config, Some(&tools)).unwrap();

        let args: HashMap<String, Value> =
            serde_json::from_str(&calls[0].function.arguments).unwrap();
        let tags = args.get("tags").unwrap().as_array().unwrap();
        assert_eq!(tags.len(), 3);
        assert_eq!(tags[0].as_str().unwrap(), "rust");
        assert_eq!(tags[1].as_str().unwrap(), "python");
        assert_eq!(tags[2].as_str().unwrap(), "go");
    }

    #[test] // helper
    fn test_type_coercion_array_json() {
        let config = get_test_config();
        let tools = vec![ToolDefinition {
            name: "tag_item".to_string(),
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": {
                    "ids": {"type": "array"}
                }
            })),
            strict: None,
        }];

        // Model emits proper JSON array
        let message = r#"<tool_call>tag_item<arg_key>ids</arg_key><arg_value>[1, 2, 3]</arg_value></tool_call>"#;
        let (calls, _) = try_tool_call_parse_glm47(message, &config, Some(&tools)).unwrap();

        let args: HashMap<String, Value> =
            serde_json::from_str(&calls[0].function.arguments).unwrap();
        let ids = args.get("ids").unwrap().as_array().unwrap();
        assert_eq!(ids.len(), 3);
        assert_eq!(ids[0].as_i64().unwrap(), 1);
    }

    #[test] // helper
    fn test_type_coercion_falls_back_to_string() {
        let config = get_test_config();
        let tools = vec![ToolDefinition {
            name: "test_func".to_string(),
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": {
                    "count": {"type": "integer"}
                }
            })),
            strict: None,
        }];

        // "not_a_number" can't be parsed as integer — should fall back to string
        let message = "<tool_call>test_func<arg_key>count</arg_key><arg_value>not_a_number</arg_value></tool_call>";
        let (calls, _) = try_tool_call_parse_glm47(message, &config, Some(&tools)).unwrap();

        let args: HashMap<String, Value> =
            serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert!(
            args.get("count").unwrap().is_string(),
            "Should fall back to string when coercion fails"
        );
    }

    /// Parser-level invariant: the glm47 parser is byte-stable — it doesn't
    /// see `finish_reason` and produces the same output regardless of the
    /// upstream stream-end reason. Real PIPELINE.finish_reason coverage (stop / tool_calls
    /// / length mapping) lives in `lib/llm/tests/test_streaming_tool_parsers.rs`
    /// and belongs in the cross-parser finish_reason mapping work-item
    /// (tracked separately).
    #[test]
    fn test_glm47_parser_output_independent_of_upstream_finish() {
        let config = get_test_config();
        let input = "<tool_call>get_weather<arg_key>location</arg_key><arg_value>NYC</arg_value></tool_call>";
        let (calls, _) = try_tool_call_parse_glm47(input, &config, None).unwrap();
        assert_eq!(calls.len(), 1);
    }

    /// TOOLCALLING.batch.9 — empty / null content variants. Truly-empty (zero bytes)
    /// and whitespace-only inputs must yield no tool calls; normal_text
    /// collapses to the empty string.
    // DEPRECATED(parser-fixture-duplicate): Duplicate of YAML fixture coverage: TOOLCALLING.batch.9 in tests/parity/toolcalling/fixtures/glm47/TOOLCALLING.batch.yaml.
    #[test] // TOOLCALLING.batch.9
    fn test_parse_glm47_empty_and_whitespace_inputs() {
        let config = get_test_config();
        for input in &["", " ", "\n", "\t\n  \t"] {
            let (calls, normal) = try_tool_call_parse_glm47(input, &config, None).unwrap();
            assert!(
                calls.is_empty(),
                "Empty/whitespace input must yield no calls (input={:?})",
                input
            );
            assert_eq!(
                normal.as_deref(),
                Some(""),
                "Empty/whitespace input collapses to empty normal_text (input={:?})",
                input
            );
        }
    }

    /// TOOLCALLING.batch.10 — duplicate calls (same function name twice in one section).
    /// Universal gap noted in the test taxonomy; pin parser-level behavior —
    /// both calls returned with distinct ids.
    // DEPRECATED(parser-fixture-duplicate): Duplicate of YAML fixture coverage: TOOLCALLING.batch.10 in tests/parity/toolcalling/fixtures/glm47/TOOLCALLING.batch.yaml.
    #[test] // TOOLCALLING.batch.10
    fn test_parse_glm47_duplicate_calls_same_name() {
        let config = get_test_config();
        let input = "<tool_call>get_weather<arg_key>location</arg_key><arg_value>NYC</arg_value></tool_call><tool_call>get_weather<arg_key>location</arg_key><arg_value>LA</arg_value></tool_call>";
        let (calls, _) = try_tool_call_parse_glm47(input, &config, None).unwrap();
        assert_eq!(calls.len(), 2, "Both duplicate-name calls must be returned");
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[1].function.name, "get_weather");
        assert_ne!(
            calls[0].id, calls[1].id,
            "Duplicate calls must have distinct ids"
        );
        let args0: HashMap<String, Value> =
            serde_json::from_str(&calls[0].function.arguments).unwrap();
        let args1: HashMap<String, Value> =
            serde_json::from_str(&calls[1].function.arguments).unwrap();
        assert_eq!(args0.get("location").unwrap().as_str().unwrap(), "NYC");
        assert_eq!(args1.get("location").unwrap().as_str().unwrap(), "LA");
    }
    #[test]
    fn null_coercion_respects_full_schema_constraints() {
        for (label, schema, expected) in [
            (
                "7-4",
                serde_json::json!({"type": ["string", "null"]}),
                serde_json::json!(null),
            ),
            (
                "7-4.anyof",
                serde_json::json!({"anyOf": [{"type": "string"}, {"type": "null"}]}),
                serde_json::json!(null),
            ),
            (
                "7-4.oneof",
                serde_json::json!({"oneOf": [{"type": "string"}, {"type": "null"}]}),
                serde_json::json!(null),
            ),
            (
                "7-4.nullable",
                serde_json::json!({"type": "string", "nullable": true}),
                serde_json::json!(null),
            ),
            (
                "7-4.const",
                serde_json::json!({"anyOf": [{"const": null}, {"type": "string"}]}),
                serde_json::json!(null),
            ),
            (
                "7-5",
                serde_json::json!({"type": "string"}),
                serde_json::json!("null"),
            ),
            (
                "7-5.union",
                serde_json::json!({"anyOf": [{"type": "string"}, {"type": "integer"}]}),
                serde_json::json!("null"),
            ),
            (
                "7-5.sibling_anyof",
                serde_json::json!({"type": "string", "anyOf": [{"type": "string"}, {"type": "null"}]}),
                serde_json::json!("null"),
            ),
            (
                "7-5.sibling_oneof",
                serde_json::json!({"type": ["string", "null"], "oneOf": [{"type": "string"}, {"type": "integer"}]}),
                serde_json::json!("null"),
            ),
            (
                "7-5.untyped_branch",
                serde_json::json!({"type": "string", "anyOf": [{"minLength": 1}, {"type": "null"}]}),
                serde_json::json!("null"),
            ),
            (
                "7-5.const",
                serde_json::json!({"anyOf": [{"const": "null"}, {"type": "integer"}]}),
                serde_json::json!("null"),
            ),
            (
                "7-5.enum",
                serde_json::json!({"type": "string", "enum": ["null", null]}),
                serde_json::json!("null"),
            ),
            (
                "allof_string",
                serde_json::json!({"allOf": [{"type": ["string", "null"]}, {"type": "string"}]}),
                serde_json::json!("null"),
            ),
            (
                "const_string",
                serde_json::json!({"const": "null"}),
                serde_json::json!("null"),
            ),
            (
                "enum_string",
                serde_json::json!({"enum": ["null"]}),
                serde_json::json!("null"),
            ),
            (
                "const_null",
                serde_json::json!({"const": null}),
                serde_json::json!(null),
            ),
            (
                "enum_null",
                serde_json::json!({"enum": [null]}),
                serde_json::json!(null),
            ),
            (
                "const_intersection",
                serde_json::json!({"type": ["string", "null"], "const": "null"}),
                serde_json::json!("null"),
            ),
            (
                "oneof_overlap",
                serde_json::json!({"oneOf": [{"type": "null"}, {}]}),
                serde_json::json!("null"),
            ),
            (
                "not_null",
                serde_json::json!({"type": ["string", "null"], "not": {"type": "null"}}),
                serde_json::json!("null"),
            ),
        ] {
            let tools = vec![ToolDefinition {
                name: "get_weather".into(),
                parameters: Some(serde_json::json!({"type":"object","properties":{"city":schema}})),
                strict: None,
            }];
            let wire = "<tool_call>get_weather<arg_key>city</arg_key><arg_value>null</arg_value></tool_call>";
            let (calls, _) =
                try_tool_call_parse_glm47(wire, &get_test_config(), Some(&tools)).unwrap();
            assert_eq!(calls.len(), 1, "{label}");
            let args: serde_json::Value =
                serde_json::from_str(&calls[0].function.arguments).unwrap();
            assert_eq!(args["city"], expected, "{label}");
        }
    }
}

#[cfg(test)]
mod enum_const_type_hint {
    use super::*;
    fn run(parameter: Value, raw: &str) -> Value {
        let tools = vec![ToolDefinition {
            name: "probe".into(),
            parameters: Some(
                serde_json::json!({"type":"object","properties":{"value": parameter}}),
            ),
            strict: None,
        }];
        let wire = format!(
            "<tool_call>probe<arg_key>value</arg_key><arg_value>{raw}</arg_value></tool_call>"
        );
        let (calls, _) =
            try_tool_call_parse_glm47(&wire, &Glm47ParserConfig::default(), Some(&tools)).unwrap();
        let arguments: Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        arguments["value"].clone()
    }
    #[test]
    fn enum_and_const_without_type_are_typed_from_their_members() {
        assert_eq!(
            run(serde_json::json!({"enum":[1,2,4,8]}), "2"),
            serde_json::json!(2)
        );
        assert_eq!(
            run(serde_json::json!({"const":3}), "3"),
            serde_json::json!(3)
        );
        assert_eq!(
            run(serde_json::json!({"enum":[1.5,2.5,3.5]}), "2.5"),
            serde_json::json!(2.5)
        );
        assert_eq!(
            run(serde_json::json!({"enum":[true,false]}), "true"),
            serde_json::json!(true)
        );
        assert_eq!(
            run(serde_json::json!({"const":true}), "true"),
            serde_json::json!(true)
        );
        assert_eq!(
            run(
                serde_json::json!({"anyOf":[{"enum":[1,2,4]},{"type":"null"}]}),
                "4"
            ),
            serde_json::json!(4)
        );
        assert_eq!(
            run(
                serde_json::json!({"anyOf":[{"enum":[1,2,4]},{"type":"null"}]}),
                "null"
            ),
            serde_json::json!(null)
        );
        // string enums stay verbatim strings, including JSON-looking text
        assert_eq!(
            run(serde_json::json!({"enum":["fast","slow"]}), "fast"),
            serde_json::json!("fast")
        );
        assert_eq!(
            run(serde_json::json!({"enum":["1","2"]}), "1"),
            serde_json::json!("1")
        );
        assert_eq!(
            run(serde_json::json!({"enum":["1","2"]}), "3"),
            serde_json::json!("3")
        );
        // a string member claims the value only when it matches, so a union's
        // numeric alternative stays reachable for non-members
        let union = serde_json::json!({"anyOf":[{"enum":["auto","none"]},{"type":"integer"}]});
        assert_eq!(run(union.clone(), "auto"), serde_json::json!("auto"));
        assert_eq!(run(union.clone(), "1"), serde_json::json!(1));
        let union = serde_json::json!({"anyOf":[{"enum":["1","2"]},{"type":"integer"}]});
        assert_eq!(run(union.clone(), "1"), serde_json::json!("1"));
        assert_eq!(run(union, "3"), serde_json::json!(3));
        // mixed enums: a non-member string yields to the numeric member
        assert_eq!(
            run(serde_json::json!({"enum":[1,"a"]}), "1"),
            serde_json::json!(1)
        );
        assert_eq!(
            run(serde_json::json!({"enum":[1,"a"]}), "a"),
            serde_json::json!("a")
        );
        // declared types unchanged
        assert_eq!(
            run(serde_json::json!({"type":["integer"]}), "30"),
            serde_json::json!(30)
        );
        assert_eq!(
            run(serde_json::json!({"type":"string"}), "7"),
            serde_json::json!("7")
        );
        // a declared string type with members claims only its members; other
        // text takes the union's typed branch (schema matrix case typed_enum_union)
        assert_eq!(
            run(
                serde_json::json!({"anyOf":[{"type":"string","enum":["auto"]},{"type":"integer"}]}),
                "42"
            ),
            serde_json::json!(42)
        );
        // an enum that contradicts the declared type does not override it
        assert_eq!(
            run(serde_json::json!({"type":"string","enum":[1,2]}), "1"),
            serde_json::json!("1")
        );
    }
}
