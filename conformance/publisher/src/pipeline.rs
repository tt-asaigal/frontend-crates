// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Parallel family resolution followed by bounded bundle assembly in one pool.
use crate::{
    codec::{self, json_bytes, sha256},
    history::{Cursor, Release, source_reference},
    resources::MemoryBudget,
    source::{self, STORE},
    yaml::{self, Node},
};
use anyhow::{Context, Result, ensure};
use rayon::prelude::*;
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::Path,
};

pub fn write(stage: &Path, relative: &str, bytes: &[u8]) -> Result<()> {
    source::relative(relative)?;
    ensure!(
        (bytes.len() as u64) < source::MAX_ARTIFACT,
        "artifact is at least 1 GB: {relative}"
    );
    let path = stage.join(relative);
    source::no_links(&path)?;
    fs::create_dir_all(path.parent().context("missing output parent")?)?;
    fs::write(path, bytes)?;
    Ok(())
}

fn descriptor(
    path: &str,
    kind: &str,
    family: Option<&str>,
    families: Option<Vec<String>>,
    capture: Option<(&str, &str)>,
    fingerprint: &str,
    catalog: &Value,
) -> Value {
    let (implementation, version) = capture.map_or((None, None), |(i, v)| (Some(i), Some(v)));
    let mut descriptor = json!({"path":path,"format":"typed-json-v1","compression":"gzip","kind":kind,"family":family});
    if let Some(families) = families {
        descriptor["families"] = json!(families);
    }
    let object = descriptor.as_object_mut().unwrap();
    object.insert("implementation".into(), json!(implementation));
    object.insert("version".into(), json!(version));
    object.insert(
        "record_contract".into(),
        implementation
            .map(|i| catalog["metadata"]["unified_capture_policy"][i]["mode"].clone())
            .unwrap_or(Value::Null),
    );
    object.insert(
        "capture_identity".into(),
        implementation
            .map(|i| json!(format!("{i}@unified@{}", version.unwrap())))
            .unwrap_or(Value::Null),
    );
    object.insert("source_fingerprint".into(), json!(fingerprint));
    descriptor
}

/// Logical envelopes have one producer; only their final encoding differs.
#[derive(Clone, Copy)]
pub enum SnapshotEncoding {
    CheckoutGzip,
    CanonicalSpool,
}

fn bundle(
    stage: &Path,
    descriptor: Value,
    typed: Value,
    metadata: Option<Value>,
    encoding: SnapshotEncoding,
) -> Result<Value> {
    let mut envelope = json!({"schema":1,"kind":descriptor["kind"],"family":descriptor["family"]});
    if let Some(families) = descriptor.get("families") {
        envelope["families"] = families.clone();
    }
    for key in ["implementation", "version", "source_fingerprint"] {
        envelope[key] = descriptor[key].clone();
    }
    envelope["value"] = typed;
    let canonical = json_bytes(&envelope)?;
    let path = descriptor["path"].as_str().context("bundle path missing")?;
    let (data, output, compression) = match encoding {
        SnapshotEncoding::CheckoutGzip => (codec::gzip(&canonical)?, path.to_string(), "gzip"),
        SnapshotEncoding::CanonicalSpool => {
            (canonical, format!(".native-envelopes/{path}"), "none")
        }
    };
    write(stage, &output, &data)?;
    let mut result = json!({"path":path,"sha256":sha256(&data),"size":data.len()});
    for (key, value) in descriptor.as_object().unwrap() {
        if key != "path" {
            result[key] = value.clone();
        }
    }
    result["compression"] = json!(compression);
    if let Some(metadata) = metadata {
        result["resolved_metadata_sha256"] = json!(sha256(&json_bytes(&metadata)?));
    }
    Ok(result)
}

fn metadata(records: &Node) -> Result<Node> {
    let mut result = Node::map();
    for (id, record) in records.mapping()? {
        let mut entry = Node::map();
        entry.set(
            "document",
            record.field("value")?.field("document")?.clone(),
        )?;
        entry.set("member", record.field("member")?.clone())?;
        result.set(id.string()?, entry)?;
    }
    Ok(result)
}

#[derive(Clone)]
struct Selection {
    implementation: String,
    version: String,
    release: Release,
    scope: Option<BTreeSet<String>>,
}
fn selections(catalog: &Value) -> Result<Vec<Selection>> {
    let mut selections = BTreeMap::new();
    for candidate in catalog["sources"]["candidates"]
        .as_array()
        .context("missing candidates")?
    {
        if candidate["mode"] != "unified" {
            continue;
        }
        let implementation = candidate["implementation"]
            .as_str()
            .context("missing implementation")?
            .to_string();
        let version = candidate["version"]
            .as_str()
            .context("missing version")?
            .to_string();
        let release = Release::parse(&version)?;
        let scope = candidate
            .get("capture_families")
            .map(|scope| {
                scope
                    .as_array()
                    .context("invalid capture scope")?
                    .iter()
                    .map(|n| Ok(n.as_str().context("invalid scoped family")?.to_string()))
                    .collect::<Result<BTreeSet<_>>>()
            })
            .transpose()?;
        source::relative(&format!("{implementation}-{version}"))?;
        ensure!(
            selections
                .insert(
                    (implementation.clone(), release.clone(), version.clone()),
                    Selection {
                        implementation,
                        version,
                        release,
                        scope
                    }
                )
                .is_none(),
            "duplicate candidate release"
        );
    }
    Ok(selections.into_values().collect())
}

fn family_source_bytes(resources: &[&Value], index: &BTreeMap<String, &Value>) -> Result<u64> {
    let mut paths = resources
        .iter()
        .map(|r| r["path"].as_str().context("missing family source path"))
        .collect::<Result<BTreeSet<_>>>()?;
    for resource in resources {
        if let Some(digest) = resource["execution"].as_str() {
            let path = format!("{STORE}/{}", source_reference(digest, "executions")?);
            let manifest = index.get(&path).context("execution is not indexed")?;
            paths.insert(
                manifest["path"]
                    .as_str()
                    .context("missing execution path")?,
            );
            for digest in manifest["receipts"]
                .as_array()
                .context("missing receipt inventory")?
            {
                let path = format!(
                    "{STORE}/{}",
                    source_reference(
                        digest.as_str().context("invalid receipt reference")?,
                        "receipts"
                    )?
                );
                let receipt = index.get(&path).context("receipt is not indexed")?;
                paths.insert(receipt["path"].as_str().context("missing receipt path")?);
            }
        }
    }
    paths.into_iter().try_fold(0u64, |sum, path| {
        sum.checked_add(
            index[path]["size"]
                .as_u64()
                .context("missing resource size")?,
        )
        .context("family dependency size overflow")
    })
}

fn source_fingerprint(resources: &[Value]) -> Result<String> {
    // Container identity belongs to source admission. Moving unchanged YAML
    // into an archive must not invalidate canonical report content identities.
    let logical = resources
        .iter()
        .map(|resource| {
            let mut resource = resource.clone();
            resource.as_object_mut().unwrap().remove("archive");
            resource
        })
        .collect::<Vec<_>>();
    Ok(sha256(&json_bytes(&json!({"resources":logical}))?))
}

/// Resolve independent families concurrently, never parallelizing a history cursor.
pub fn snapshots(
    root: &Path,
    stage: &Path,
    catalog: &Value,
    budget: &MemoryBudget,
    encoding: SnapshotEncoding,
) -> Result<Value> {
    let original = catalog["sources"]["unified"]
        .as_array()
        .context("missing Unified sources")?;
    if original.is_empty() {
        return Ok(catalog.clone());
    }
    let fingerprint = source_fingerprint(original)?;
    let selected = selections(catalog)?;
    let mut families = BTreeMap::<String, Vec<&Value>>::new();
    for resource in original {
        if let Some(family) = resource["family"].as_str() {
            families
                .entry(family.to_string())
                .or_default()
                .push(resource);
        }
    }
    let index = original
        .iter()
        .map(|r| {
            Ok((
                r["path"]
                    .as_str()
                    .context("missing source path")?
                    .to_string(),
                r,
            ))
        })
        .collect::<Result<BTreeMap<_, _>>>()?;
    let parts = stage.join(".native-parts");
    fs::create_dir(&parts)?;
    let per_family = families
        .par_iter()
        .map(|(name, resources)| -> Result<Value> {
            let source_bytes = family_source_bytes(resources, &index)?;
            let reserved = source_bytes
                .checked_mul(32)
                .context("family reservation overflow")?
                .max(16 * 1024 * 1024);
            let _reservation = budget.reserve(reserved)?;
            let authored_resource = resources
                .iter()
                .find(|r| r["kind"] == "inputs-and-golden")
                .context("missing authored family")?;
            ensure!(
                authored_resource["path"]
                    == format!("{STORE}/families/{name}/inputs_and_golden.yaml"),
                "family path mismatch"
            );
            let bytes = source::verified(root, authored_resource)?;
            let authored = yaml::load_str(std::str::from_utf8(&bytes)?)?;
            source::family(&authored, name, &catalog["metadata"]["unified_aliases"])
                .with_context(|| name.clone())?;
            let input_descriptor = descriptor(
                &format!("conformance/viewer-data/{name}/inputs-and-golden.json.gz"),
                "inputs-and-golden",
                Some(name),
                None,
                None,
                &fingerprint,
                catalog,
            );
            let typed_authored = authored.typed();
            let input = bundle(stage, input_descriptor, typed_authored, None, encoding)?;
            let mut implementations = BTreeMap::<String, Vec<&Value>>::new();
            for resource in resources {
                if let Some(implementation) = resource["implementation"].as_str() {
                    implementations
                        .entry(implementation.to_string())
                        .or_default()
                        .push(resource);
                }
            }
            for (implementation, resources) in implementations {
                let history = source::load_history(
                    root,
                    &authored,
                    name,
                    &implementation,
                    &resources,
                    catalog["metadata"]["unified_capture_policy"][&implementation]["mode"]
                        == "batch",
                    source::HistoryAdmission {
                        limit: reserved / 2,
                        index: &index,
                    },
                )
                .with_context(|| format!("{name}/{implementation}"))?;
                let releases = history
                    .captures
                    .iter()
                    .map(|c| Release::parse(&c.version))
                    .collect::<Result<Vec<_>>>()?;
                let mut cursor = Cursor::default();
                let mut position = 0;
                for selection in selected
                    .iter()
                    .filter(|s| s.implementation == implementation)
                {
                    while position < releases.len() && releases[position] <= selection.release {
                        cursor.apply(&history, position)?;
                        position += 1;
                    }
                    if position == 0 || selection.scope.as_ref().is_some_and(|s| !s.contains(name))
                    {
                        continue;
                    }
                    let records = cursor.records(&history, &selection.version)?;
                    let part =
                        json!({"value":records.typed(),"metadata":metadata(&records)?.typed()});
                    write(
                        stage,
                        &format!(
                            ".native-parts/{implementation}-{}/{name}.json",
                            selection.version
                        ),
                        &json_bytes(&part)?,
                    )?;
                }
            }
            Ok(input)
        })
        .collect::<Result<Vec<_>>>()?;
    let mut resources = per_family;
    let captures = selected
        .par_iter()
        .map(|selection| -> Result<Option<Value>> {
            let path = parts.join(format!(
                "{}-{}",
                selection.implementation, selection.version
            ));
            if !path.exists() {
                return Ok(None);
            }
            let mut paths = fs::read_dir(&path)?
                .map(|p| Ok(p?.path()))
                .collect::<Result<Vec<_>>>()?;
            paths.sort();
            let source_bytes = paths
                .iter()
                .map(|p| fs::metadata(p).map(|m| m.len()))
                .collect::<std::io::Result<Vec<_>>>()?
                .into_iter()
                .sum::<u64>();
            let _reservation = budget.reserve(
                source_bytes
                    .checked_mul(20)
                    .context("bundle reservation overflow")?
                    .max(16 * 1024 * 1024),
            )?;
            let mut values = Vec::new();
            let mut metadata = Vec::new();
            let mut names = Vec::new();
            for path in paths {
                let name = path
                    .file_stem()
                    .and_then(|n| n.to_str())
                    .context("invalid fragment")?;
                let mut part: Value = serde_json::from_slice(&fs::read(&path)?)?;
                // Each private fragment has one consumer; release it before envelopes accumulate.
                fs::remove_file(&path)?;
                values.push(json!([["str", name], part["value"].take()]));
                metadata.push(json!([["str", name], part["metadata"].take()]));
                names.push(name.to_string());
            }
            fs::remove_dir(&path)?;
            let descriptor = descriptor(
                &format!(
                    "conformance/viewer-data/bundles/{}-{}.json.gz",
                    selection.implementation, selection.version
                ),
                "resolved-capture",
                None,
                Some(names),
                Some((&selection.implementation, &selection.version)),
                &fingerprint,
                catalog,
            );
            Ok(Some(bundle(
                stage,
                descriptor,
                json!(["object", values]),
                Some(json!(["object", metadata])),
                encoding,
            )?))
        })
        .collect::<Result<Vec<_>>>()?;
    resources.extend(captures.into_iter().flatten());
    resources.sort_by_key(|r| {
        (
            if r["kind"] == "inputs-and-golden" {
                0
            } else {
                1
            },
            r["family"]
                .as_str()
                .or(r["implementation"].as_str())
                .unwrap_or("")
                .to_string(),
            r["version"].as_str().unwrap_or("").to_string(),
        )
    });
    let mut result = catalog.clone();
    result["source_revision"] = catalog["revision"].clone();
    result["sources"]["original_unified"] = json!(original);
    result["sources"]["unified"] = json!(resources);
    result.as_object_mut().unwrap().remove("revision");
    result["revision"] = json!(sha256(&json_bytes(&result)?));
    fs::remove_dir_all(parts)?;
    Ok(result)
}

pub fn projection(catalog: &Value) -> Result<Value> {
    let keys = [
        "selection_policy",
        "unavailable_categories",
        "implementations",
        "unified_display",
        "unified_capture_policy",
        "markers",
        "unified",
        "unified_presentation",
        "folds",
        "unified_taxonomy",
        "unified_aliases",
        "unified_leak_markers",
        "documentation",
        "peer_versions",
    ];
    let mut metadata = json!({});
    for key in keys {
        if let Some(value) = catalog["metadata"].get(key) {
            metadata[key] = value.clone();
        }
    }
    let mut legacy = json!({});
    let slug = regex::Regex::new("[^a-zA-Z0-9]+")?;
    for candidate in catalog["sources"]["candidates"]
        .as_array()
        .context("missing candidates")?
    {
        let mode = candidate["mode"]
            .as_str()
            .context("missing candidate mode")?;
        if !["batch", "stream", "batch-on-stream", "unified"].contains(&mode) {
            continue;
        }
        let implementation = candidate["implementation"]
            .as_str()
            .context("missing candidate implementation")?;
        let version = candidate["version"]
            .as_str()
            .context("missing candidate version")?;
        let mode_key = if mode == "batch" {
            "b-"
        } else if mode == "batch-on-stream" {
            "s-"
        } else {
            ""
        };
        legacy[format!(
            "{implementation}-{mode_key}{}",
            slug.replace_all(version, "-").trim_matches('-')
        )] = json!(format!("{implementation}@{version}"));
    }
    metadata["legacy_candidate_identities"] = legacy;
    metadata["tabs"] = json!(
        catalog["metadata"]["tabs"]
            .as_array()
            .context("missing tabs")?
            .iter()
            .filter(|t| t["suite"] == "unified")
            .collect::<Vec<_>>()
    );
    metadata["selection_policy"] = json!({"unified":metadata["selection_policy"].get("unified").cloned().unwrap_or(json!({}))});
    if metadata.get("folds").is_none() {
        metadata["folds"] = json!({});
    }
    let mut result = json!({});
    if let Some(revision) = catalog.get("source_revision") {
        result["source_revision"] = revision.clone();
    }
    for (key, value) in [
        ("schema", catalog.get("schema").cloned().unwrap_or(json!(1))),
        ("revision", catalog["revision"].clone()),
        ("snapshot", catalog["snapshot"].clone()),
        ("metadata", metadata),
    ] {
        result[key] = value;
    }
    let mut sources = json!({});
    if let Some(original) = catalog["sources"].get("original_unified") {
        sources["original_unified"] = original.clone();
    }
    sources["archives"] = json!([]);
    sources["unified"] = catalog["sources"]["unified"].clone();
    sources["candidates"] = json!(
        catalog["sources"]["candidates"]
            .as_array()
            .context("missing candidates")?
            .iter()
            .filter(|c| c["mode"] == "unified")
            .collect::<Vec<_>>()
    );
    result["sources"] = sources;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn source_identity_depends_on_yaml_bytes_not_archive_encoding() -> Result<()> {
        let original = json!({"path":"families/sample/dynamo_v2-0.1.0.yaml","size":10,"sha256":"a".repeat(64)});
        let mut packed = original.clone();
        packed["archive"] =
            json!({"path":"versions/dynamo-0.1.0.tar.gz","size":100,"sha256":"b".repeat(64)});
        assert_eq!(
            source_fingerprint(std::slice::from_ref(&original))?,
            source_fingerprint(std::slice::from_ref(&packed))?
        );
        packed["sha256"] = json!("c".repeat(64));
        assert_ne!(
            source_fingerprint(&[original])?,
            source_fingerprint(&[packed])?
        );
        Ok(())
    }
    #[test]
    fn canonical_spool_and_checkout_gzip_share_one_envelope() -> Result<()> {
        let stage = crate::transaction::unique(&std::env::temp_dir(), "envelope-encoding-")?;
        let result = (|| -> Result<()> {
            let descriptor = descriptor(
                "conformance/viewer-data/family/inputs-and-golden.json.gz",
                "inputs-and-golden",
                Some("family"),
                None,
                None,
                "fingerprint",
                &json!({}),
            );
            let typed = json!(["object", []]);
            let metadata = Some(json!(["object", []]));
            let canonical = bundle(
                &stage,
                descriptor.clone(),
                typed.clone(),
                metadata.clone(),
                SnapshotEncoding::CanonicalSpool,
            )?;
            let checkout = bundle(
                &stage,
                descriptor,
                typed,
                metadata,
                SnapshotEncoding::CheckoutGzip,
            )?;
            let bytes = source::verified(&stage.join(".native-envelopes"), &canonical)?;
            let compressed = source::verified(&stage, &checkout)?;
            assert_eq!(canonical["compression"], "none");
            assert_eq!(checkout["compression"], "gzip");
            assert_eq!(
                bytes,
                codec::gunzip(
                    &compressed,
                    checkout["sha256"].as_str().unwrap(),
                    bytes.len()
                )?
            );
            assert_eq!(
                canonical["resolved_metadata_sha256"],
                checkout["resolved_metadata_sha256"]
            );
            assert!(!stage.join(".native-parts/inputs").exists());
            assert!(
                !stage
                    .join("conformance/viewer-data/bundles/inputs-and-golden.json.gz")
                    .exists()
            );
            Ok(())
        })();
        fs::remove_dir_all(stage)?;
        result
    }

    #[test]
    fn compact_dependency_bytes_are_part_of_family_admission() -> Result<()> {
        let pointer =
            json!({"path":"family/pointer.yaml","size":1,"execution":"executions/abc.yaml"});
        let execution = json!({"path":format!("{STORE}/executions/abc.yaml"),"size":1_000_000,"receipts":["receipts/def.yaml"]});
        let receipt = json!({"path":format!("{STORE}/receipts/def.yaml"),"size":2_000_000});
        let index = [&pointer, &execution, &receipt]
            .into_iter()
            .map(|r| (r["path"].as_str().unwrap().to_string(), r))
            .collect();
        assert_eq!(family_source_bytes(&[&pointer], &index)?, 3_000_001);
        let manifest = yaml::load_str(
            "schema_version: 1\nfamily: sample\nimplementation: demo\nversion: 1.0.0\ncheckpoint: {provenance: {}, document: {}, document_overrides: {}}\ncases: [one, two]\ndefaults: {metadata: expansion}\nexceptions: {}\n",
        )?;
        assert!(
            crate::history::expand_execution_bounded(
                &manifest,
                "sample",
                "demo",
                "1.0.0",
                &BTreeMap::new(),
                1
            )
            .is_err()
        );
        Ok(())
    }
}
