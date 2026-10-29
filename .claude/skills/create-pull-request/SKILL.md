---
name: create-pull-request
description: Create and publish a GitHub pull request for the current branch with a title and description that follow this project's conventions (`AB-{ID}: Title`, template from .github/pull_request_template.md). Use when the user asks to open, create, or publish a PR / pull request.
allowed-tools: Bash(git:*), Bash(gh:*)
---

Open a PR on GitHub (`Juanpacol/Tumor-Classification-App`) following `CONTRIBUTING.md`. PRs live in GitHub, NOT Azure DevOps (Azure Boards is only for tickets).

## Target branch

Every PR targets **`develop`** first. Only a **hotfix** targets `main`.

- A PR is a hotfix only if the user says so (e.g. "hotfix", "urgent production fix") or the branch is named `AB-{ID}-hotfix-{description}`. Never infer a hotfix from the commit type alone (`fix` is not a hotfix).
- If it is a hotfix: base = `main`, and prefix the title description with `[HOTFIX]` (e.g. `AB-120: [HOTFIX] Fix upload crash`). Remind the user that the fix must also be merged back into `develop` afterward.
- If it is ambiguous whether the change is a hotfix, ask; default to `develop`.

Below, `<base>` means `develop` or `main` according to these rules.

## Steps

1. Gather context in parallel: `git fetch origin`, `git branch --show-current`, `git status --short`, `git log origin/<base>..HEAD --oneline`, `git diff origin/<base>...HEAD --stat`, `gh auth status`. Decide `<base>` first (see above).
2. Preconditions:
   - Branch must be `AB-{ID}-{description}` and not `main` or `develop`; otherwise stop and ask. Branches should be created from `develop` (from `main` for hotfixes).
   - Uncommitted changes: tell the user and offer to run the `create-commit` skill first.
   - Every commit subject must match `^AB-[0-9]+: (feat|fix|docs|refactor|test|chore)\([a-z_]+\):.+` (the `validate-commits` CI check). Flag any that don't before publishing.
   - Check the PR doesn't already exist: `gh pr view --json url 2>/dev/null`; if it does, report its URL instead of creating another.
3. Draft the content, **in English**:
   - **Title**: `AB-{ID}: Short description` (capitalized, no type/scope prefix, no trailing period), derived from the branch name and commits.
   - **Body**: read `.github/pull_request_template.md` and fill it, keeping the required sections:
     - `## What`: 1-3 sentences on what changed and why (from the full diff, not just the last commit).
     - `## How to test`: concrete commands/steps (`cd app/backend && pytest`, `npm run dev`, etc.) relevant to the changed scopes.
     - `## Related ticket`: `Closes #{ID}`.
     - `## Screenshots / links`: fill only if applicable, otherwise remove the section.
     - Remove the HTML comment instructions.
     - End with `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.
4. Show the title and body to the user and ask for confirmation before publishing, since a PR is outward-facing. Skip the confirmation only if the user explicitly said to publish without asking.
5. Push if needed: `git push -u origin <branch>` (never force-push).
6. Write the body to a temp file and create the PR against `<base>`:
   ```bash
   gh pr create --base <base> --head <branch> --title "AB-116: Title" --body-file <tmpfile>
   ```
7. Return the PR URL and the base branch used. Remind the user: merge strategy is **Rebase and merge** only, at least 1 review is required, and the Azure Boards work item must be closed manually after merge (`Closes #ID` does not close it).

## Rules

- Never merge the PR, and never use `--no-verify` or force flags.
- Everything written to GitHub (title, body) must be English.
- Never target `main` for a non-hotfix PR; the `develop` -> `main` promotion is a separate, explicit request.
- Add `--draft` only if the user asks.
