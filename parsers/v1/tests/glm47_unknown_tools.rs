// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_parsers::tool_calling::ToolDefinition;
use dynamo_parsers::tool_calling::jail::{
    Annotated, apply_tool_calling_jail, apply_tool_calling_jail_with_guided_streaming,
};
use dynamo_protocols::types::{
    ChatCompletionMessageContent, ChatCompletionToolChoiceOption,
    CreateChatCompletionStreamResponse, FinishReason,
};
use futures::{StreamExt, stream};

fn chunk(
    text: &str,
    finish: Option<FinishReason>,
) -> Annotated<CreateChatCompletionStreamResponse> {
    Annotated {
        data: Some(
            serde_json::from_value(serde_json::json!({
                "id": "synthetic", "object": "chat.completion.chunk", "created": 0,
                "model": "synthetic", "choices": [{
                    "index": 0, "delta": {"content": text}, "finish_reason": finish
                }]
            }))
            .unwrap(),
        ),
        id: None,
        event: None,
        comment: None,
        error: None,
    }
}

async fn run(
    wire: &str,
    width: usize,
    choice: ChatCompletionToolChoiceOption,
    structural_tag: bool,
) -> (Vec<String>, String, Vec<FinishReason>) {
    let streamed = run_with_options(
        wire,
        width,
        choice.clone(),
        structural_tag,
        true,
        FinishReason::Stop,
    )
    .await;
    let buffered = run_with_options(
        wire,
        width,
        choice,
        structural_tag,
        false,
        FinishReason::Stop,
    )
    .await;
    assert_eq!(
        streamed, buffered,
        "guided streaming and buffering must agree"
    );
    streamed
}

async fn run_with_options(
    wire: &str,
    width: usize,
    choice: ChatCompletionToolChoiceOption,
    structural_tag: bool,
    guided_streaming: bool,
    terminal: FinishReason,
) -> (Vec<String>, String, Vec<FinishReason>) {
    let mut input: Vec<_> = wire
        .chars()
        .collect::<Vec<_>>()
        .chunks(width)
        .map(|part| chunk(&part.iter().collect::<String>(), None))
        .collect();
    input.push(chunk("", Some(terminal)));
    let tools = Some(vec![ToolDefinition {
        name: "read".into(),
        parameters: None,
        strict: None,
    }]);
    let output: Vec<_> = if guided_streaming {
        apply_tool_calling_jail_with_guided_streaming(
            Some("glm47".into()),
            Some(choice),
            tools,
            structural_tag,
            stream::iter(input),
        )
        .collect()
        .await
    } else {
        apply_tool_calling_jail(
            Some("glm47".into()),
            Some(choice),
            tools,
            structural_tag,
            stream::iter(input),
        )
        .collect()
        .await
    };
    let mut names = Vec::new();
    let mut text = String::new();
    let mut finishes = Vec::new();
    let mut arguments = std::collections::BTreeMap::<u32, String>::new();
    for frame in output {
        for choice in frame.data.unwrap().choices {
            if let Some(ChatCompletionMessageContent::Text(content)) = choice.delta.content {
                text.push_str(&content);
            }
            for call in choice.delta.tool_calls.into_iter().flatten() {
                let function = call.function.unwrap();
                if let Some(name) = function.name {
                    assert!(
                        arguments.insert(call.index, String::new()).is_none(),
                        "duplicate tool opener"
                    );
                    names.push(name);
                }
                arguments
                    .get_mut(&call.index)
                    .expect("argument delta without a tool opener")
                    .push_str(function.arguments.as_deref().unwrap_or(""));
            }
            finishes.extend(choice.finish_reason);
        }
    }
    for argument in arguments.values() {
        assert!(
            serde_json::from_str::<serde_json::Value>(argument).is_ok(),
            "broken argument stream: {argument}"
        );
    }
    (names, text, finishes)
}

#[tokio::test]
async fn required_native_fallback_drops_unknown_markup_and_keeps_prose() {
    for structural_tag in [false, true] {
        for width in [1, 7, 1024] {
            let (names, text, finishes) = run(
                "Before. <tool_call>unknown<arg_key>x</arg_key><arg_value>1</arg_value></tool_call>",
                width, ChatCompletionToolChoiceOption::Required, structural_tag,
            ).await;
            assert!(names.is_empty());
            assert_eq!(
                text, "Before. ",
                "width={width}, structural_tag={structural_tag}"
            );
            assert_eq!(finishes, [FinishReason::Stop]);
        }
    }
}

#[tokio::test]
async fn guided_json_drops_unknown_calls_before_any_delta() {
    for wire in [
        r#"[{"name":"unknown","arguments":{"x":1}}]"#,
        r#"[{"name":"unknown","arguments":{"x":"unfinished"#,
    ] {
        for width in [1, 7, 1024] {
            let (names, text, finishes) =
                run(wire, width, ChatCompletionToolChoiceOption::Required, false).await;
            assert!(names.is_empty(), "width={width}, names={names:?}");
            assert!(text.is_empty(), "width={width}, content={text:?}");
            assert_eq!(finishes, [FinishReason::Stop]);
        }
    }
}

#[tokio::test]
async fn guided_json_keeps_known_calls_after_unknown_calls() {
    let wire = r#"[{"name":"unknown","arguments":{"x":1}},{"name":"read","arguments":{"x":2}}]"#;
    for width in [1, 7, 1024] {
        let (names, text, finishes) =
            run(wire, width, ChatCompletionToolChoiceOption::Required, false).await;
        assert_eq!(names, ["read"], "width={width}");
        assert!(text.is_empty());
        assert_eq!(finishes, [FinishReason::ToolCalls]);
    }
}

#[tokio::test]
async fn bare_unknown_call_keeps_prefix_without_releasing_markup() {
    for width in [1, 7, 1024] {
        let (names, text, finishes) = run(
            "Before. unknown<arg_key>x</arg_key><arg_value>1</arg_value></tool_call>",
            width,
            ChatCompletionToolChoiceOption::Auto,
            false,
        )
        .await;
        assert!(names.is_empty());
        assert_eq!(text, "Before. ");
        assert_eq!(finishes, [FinishReason::Stop]);
    }
}

#[tokio::test]
async fn bare_unknown_name_with_known_suffix_is_not_renamed() {
    for width in [1, 7, 1024] {
        let (names, text, _) = run(
            "Before. mcp__repo__read<arg_key>x</arg_key><arg_value>1</arg_value></tool_call>",
            width,
            ChatCompletionToolChoiceOption::Auto,
            false,
        )
        .await;
        assert!(names.is_empty(), "width={width}, names={names:?}");
        assert_eq!(text, "Before. ");
    }
}

#[tokio::test]
async fn glm47_prose_is_preserved_while_holding_possible_bare_names() {
    for wire in [
        "Please read this. ",
        "read",
        "read ",
        "mcp__repo__read",
        "Hello 世界!",
        "  ",
    ] {
        for width in [1, 7, 1024] {
            for terminal in [FinishReason::Stop, FinishReason::Length] {
                let (names, text, finishes) = run_with_options(
                    wire,
                    width,
                    ChatCompletionToolChoiceOption::Auto,
                    false,
                    true,
                    terminal,
                )
                .await;
                assert!(names.is_empty());
                assert_eq!(text, wire);
                assert_eq!(finishes, [terminal]);
            }
        }
    }
}

#[tokio::test]
async fn named_native_fallback_and_length_finish_drop_unknown_calls() {
    let named: ChatCompletionToolChoiceOption = serde_json::from_value(serde_json::json!({
        "type": "function", "function": {"name": "read"}
    }))
    .unwrap();
    for structural_tag in [false, true] {
        for guided in [false, true] {
            let (names, text, finishes) = run_with_options(
                "Before. <tool_call>unknown<arg_key>x</arg_key><arg_value>1</arg_value></tool_call>",
                1, named.clone(), structural_tag, guided, FinishReason::Length,
            ).await;
            assert!(names.is_empty());
            assert_eq!(text, "Before. ");
            assert_eq!(finishes, [FinishReason::Length]);
        }
    }
}
