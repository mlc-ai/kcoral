# AI pull request review

## Selected setup

Use CodeRabbit as an advisory reviewer for KCoral, configured in
[`../.coderabbit.yaml`](../.coderabbit.yaml). Installation is separate from this
file: an `mlc-ai` organization owner must approve the GitHub App for **only
`mlc-ai/kcoral`**. A committed configuration does not prove the app is installed
or that a review has run.

The configuration reviews non-draft PRs targeting the default branch and their
subsequent pushes. It pauses after five reviewed commits; a maintainer can request
another pass with `@coderabbitai review` or resume with `@coderabbitai resume`.
Titles containing `[skip ai review]` opt out of automatic reviews.

Reviews focus on Python runtime/server/client behavior, GPU leases, sandbox and
process cleanup, Rust routing, protocol compatibility, and workflow permissions.
Lockfiles, generated Python protobuf bindings, and brand images are excluded;
dependency manifests and the source protobuf schema remain reviewable.

Automatic approval, label/reviewer assignment, issue planning, and the configured
code-writing finishing touches are disabled. Reviews use English to match the
repository. Chat requires an explicit mention and is limited to organization
members; automatic reviews still cover external contributors. Knowledge-base
retention features, web search, and automatic cross-repository linking are off.
These application settings are **not a GitHub permission boundary**: CodeRabbit
reads the configuration in the PR branch, and the App retains its granted rights.

## Existing integration audit (2026-10-05)

- KCoral is a public Python/Rust repository. The default branch is `main`.
- The active `Main Protection` ruleset already requests Copilot review on new
  PRs and pushes; draft reviews are disabled. The settings remain unchanged.
- Copilot returned a quota-limit message instead of a review on
  [PR #114](https://github.com/mlc-ai/kcoral/pull/114) and
  [PR #116](https://github.com/mlc-ai/kcoral/pull/116). An automatic trigger alone
  therefore does not currently guarantee coverage.
- Current workflow files build wheels, build/publish documentation, and publish
  releases to PyPI. GitHub also lists an older gateway test workflow and dynamic
  Copilot, dependency graph, and Pages workflows. No workflow is changed here.
- Installed Apps at inspection: ChatGPT Codex Connector, mlc-ai-jenkins,
  Review Notebook App, and Slack. No CodeRabbit, Qodo, Greptile, or Gemini App
  was listed for this repository. Early PRs contain Gemini review comments.
- The configuring account has repository admin rights but is an organization
  member, so GitHub offers an installation request rather than direct install.
- GitHub detects no repository license and there is no root LICENSE file.
  `rust/kcoral/Cargo.toml` declares Apache-2.0 for that package. This change does
  not assign a license to the rest of the repository.

## Alternatives considered

Policies below were checked against official sources on 2026-10-05; they may change.
This comparison concerns integration fit, not a measured accuracy benchmark.

| Tool | Automatic / inline review | Free or OSS policy | Access and configuration | Decision |
| --- | --- | --- | --- | --- |
| CodeRabbit | New PRs and incremental push reviews; inline findings and suggestions | OSS access includes Team features for public repositories, with adaptive limits; the ordinary Free tier only summarizes PRs | Selected-repository GitHub App; code, workflow, checks, statuses, issues and PR write access; `.coderabbit.yaml` with path rules | Selected for managed operation, OSS access and explicit controls; broad App rights require owner review |
| GitHub Copilot | Native ruleset triggers and inline suggestions; already configured here | Reviews consume AI credits; public-repo Actions minutes being free does not make model usage free | Existing native integration; repository instructions and rulesets | Preserve existing configuration; current quota failures make it insufficient alone |
| Qodo (formerly Qodo Merge) | Automated PR review and configurable inline findings | Eligible public projects can receive free core reviews; 200+ stars on this repo **or another public repo in the organization**; advanced Rules/PR History are paid | GitHub App scoped to selected repositories; repository sync and PR indexing; `.pr_agent.toml` or portal | Viable backup: mlc-ai/mlc-llm satisfies the organization star condition; other eligibility checks still apply |
| Greptile | Automatic PR review and inline suggestions with repository context | Free OSS program requires qualification; the program page says OSI-approved license, while pricing mentions MIT/Apache noncommercial projects | GitHub integration with codebase indexing; directory `.greptile/` configuration or `greptile.json`; review requested App scopes before installing | Defer until repository licensing and free-program acceptance are clear |
| Gemini Code Assist on GitHub | Enterprise version supports PR-open reviews and inline suggestions; `/gemini review` for another pass | Consumer GitHub app shut down on 2026-07-17. Enterprise preview reviews have no charge, but require a billing-enabled Cloud project; Developer Connect can incur usage charges | Cloud IAM and Developer Connect plus GitHub access; `.gemini/config.yaml` and `styleguide.md`; workflow files are excluded from suggestions | Not the simplest free installation; no Cloud resources or billing configured here |

Sources: [CodeRabbit plans](https://docs.coderabbit.ai/management/plans),
[automatic triggers](https://docs.coderabbit.ai/configuration/auto-review),
[GitHub permissions](https://docs.coderabbit.ai/platforms/github-com),
[configuration](https://docs.coderabbit.ai/reference/configuration),
[Copilot usage](https://docs.github.com/en/copilot/concepts/agents/code-review),
[Qodo OSS](https://docs.qodo.ai/open-source-program),
[Qodo configuration](https://docs.qodo.ai/configuration/configuration-file),
[Qodo installation](https://docs.qodo.ai/install-qodo/github/qodo-multi-tenant),
[Greptile OSS](https://www.greptile.com/open-source),
[Greptile pricing](https://www.greptile.com/pricing),
[Greptile configuration](https://www.greptile.com/docs/code-review/greptile-json-reference),
[Gemini consumer sunset](https://developers.google.com/gemini-code-assist/docs/deprecations/consumer-code-review),
[Gemini enterprise use and billing](https://docs.cloud.google.com/gemini/docs/code-review/use-code-assist-github),
[Gemini configuration](https://docs.cloud.google.com/gemini/docs/code-review/customize-repo-review).

## Installation and verification

1. Review the [official App](https://github.com/apps/coderabbitai), select the
   `mlc-ai` organization, choose **Only select repositories**, and select
   **kcoral**. An organization owner must approve installation.
2. Review the actual authorization screen. At inspection it requested read access
   to Actions, discussions, members, merge queues, and metadata, plus write access
   to checks, code, commit statuses, issues, PRs, and workflows. Review-only use
   still grants these broader permissions; limiting repository scope is essential.
3. Confirm public/OSS review entitlement in CodeRabbit. Do not mistake the
   time-limited trial or summary-only Free tier for permanent OSS reviews. Do not
   enable paid usage add-ons as part of this setup.
4. Merge the configuration PR through the normal review process. No Actions
   workflow, API-key secret, required check, or branch-protection change is needed.
5. Open a non-draft PR targeting `main`, or explicitly request review on the
   configuration PR. Confirm a completed CodeRabbit review and absence of YAML
   errors. A clean PR may correctly produce no inline findings; do not inject
   defects just to manufacture comments. Push a real follow-up change and confirm
   an incremental review. Check skip/draft behavior when those cases occur.
6. Use `@coderabbitai configuration` to inspect the effective settings. If review
   fails, check installation scope, OSS entitlement and review rate limits first.

Keep human review and existing CI as the basis for merge decisions. CodeRabbit's
review status is not added as a required check. Existing Copilot auto-review is
unchanged, so both may comment after its quota recovers.

## Pause or remove

- Per PR: `@coderabbitai pause`; resume with `@coderabbitai resume`.
- Repository automatic reviews: change `reviews.auto_review.enabled` to `false`
  through a PR. Manual review commands remain available while the App is installed.
- Revoke access: an organization owner removes kcoral from the App's selected
  repositories or uninstalls the App if unused. Deleting the YAML alone does not
  disable the service; its defaults can resume automatic reviews.
