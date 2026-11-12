# Tenstorrent patches on dynamo-parsers

This fork carries the GLM-4.7 and Kimi K3 tool-call argument fixes the Tenstorrent stack needs, on top of the exact
`dynamo-parsers` release the stack's ai-dynamo wheel pins. Nothing here tracks upstream `main`.

## Branches and tags

| Branch | Base tag | Tag | Serves |
| --- | --- | --- | --- |
| `tt/parsers-10.0.7` | `dynamo-parsers-v10.0.7` | `dynamo-parsers-v10.0.7+tt.2` | ai-dynamo 1.6.0, 1.6.1, 1.6.2 |
| `tt/parsers-10.2.2` | `dynamo-parsers-v10.2.2` | `dynamo-parsers-v10.2.2+tt.2` | ai-dynamo 1.7.0 pre-releases |
| `tt/parsers-11.0.11` | `dynamo-parsers-v11.0.11` | `dynamo-parsers-v11.0.11+tt.2` | whatever pins 11.x next |

A `+tt.N` tag is cut only after `cargo test -p dynamo-parsers` is green on that branch. `+tt.1` carried patches 1 to 4;
`+tt.2` adds patches 5 and 6.

## The patches

Patches 1 to 4 are in `parsers/v1/src/tool_calling/xml/glm47_parser.rs` (plus two expected values in
`conformance/utils/src/glm47_schema_cases.json`); 5 and 6 also touch the streaming jail (`parsers/v1/src/tool_calling/jail/`).
Patch 7 is in `parsers/v1/src/tool_calling/xtml/kimi_k3_parser.rs` and reuses patch 4's helpers from the GLM-4.7 parser.

| # | Patch | Upstream status |
| --- | --- | --- |
| 1 | Derive the argument type hint from enum/const members | Merged as #439 (Nov 3). Carried on 10.0.7 and 10.2.2, which predate it; dropped on 11.0.11. |
| 2 | Match type-less string literals exactly; a declared type keeps its hint | Not submitted. On 11.0.11 it replaces #439's block. |
| 3 | Resolve parameter schemas through root combinators, prefer parsed scalars over bare string branches, honour not/if-then-else and untyped schemas | Not submitted. Upstream's #517 did part of this for the shared XML parser, not for GLM-4.7. |
| 4 | Coerce quoted scalars where the schema admits no string; fill omitted required nullable keys with null | Not submitted. |
| 5 | Drop tool calls to functions the request did not offer (fork PR #1, ksriv001) | Not submitted; reverses upstream #263, which passes such calls through so the client can correct the model. Our stack has no executor-side check and scorers count each one as an unknown name. Matches vLLM's `validate_tool_names`. |
| 6 | Close the remaining unknown-tool paths in the stream jail; content-free logs (fork PR #1) | Not submitted. With a tools list present, the jail holds a chunk's trailing identifier until the next chunk shows it is not a bare call name: up to one chunk of extra latency on prose, released at end of stream. On 11.0.11 the wrapper takes upstream's prose-line indent. |
| 7 | kimi_k3 parser: patch 4's two rules for Kimi K3 JSON arguments | Not submitted. On 10.0.7 only. |

On 11.0.11 patch 1 is absent (upstream has it): patch 2 as re-applied there also keeps upstream's own changes to the
file since 10.0.7 (`has_unsupported_schema_ref_scope` is `pub(crate)` for #517; whitespace between `</arg_key>` and
`<arg_value>` per #472, with its test), then patches 3 to 6.

## Where we disagree with upstream

Two expected values differ from upstream's, and the branches keep ours. Each needs an upstream PR with the case,
or the gap reopens at every uplift.

| Schema | Raw | Ours | Upstream |
| --- | --- | --- | --- |
| `{"enum": [1, "a"]}` | `1` | `1` (the integer member) | `"1"` (string fallback, not a member) |
| `{"type": ["string", "integer"]}` | `42` | `42` | `"42"` |
| A call to a function not in the request's `tools` | | dropped, prose kept | returned to the client (#263); the conformance capture `glm47/TOOLCALLING.batch.13` still encodes this |

## Uplift procedure

1. Read the `dynamo-parsers` pin in ai-dynamo's `Cargo.toml` at the release the stack is moving to.
2. `git checkout -b tt/parsers-<pin> dynamo-parsers-v<pin>`, then cherry-pick the patches from the nearest `tt/` branch.
   Patches already merged upstream are dropped. Conflicts mean upstream reworked the same code; resolve toward
   the behaviour in the table above and re-check the conformance fixture.
3. `cargo test -p dynamo-parsers`, then the stack's frontend tests against a wheel built with the branch.
4. Tag `dynamo-parsers-v<pin>+tt.1` and point the stack's frontend build at it.

## How the stack consumes a branch

The ai-dynamo wheel pins `dynamo-parsers` from crates.io, so the stack builds the wheel with

```toml
[patch.crates-io]
dynamo-parsers = { git = "https://github.com/tt-asaigal/frontend-crates", tag = "dynamo-parsers-v10.0.7+tt.1" }
```

in ai-dynamo's workspace `Cargo.toml`. The patched crate must carry the same version as the pin.
