# Windows release signing

The workflow `.github/workflows/windows-signed-release.yml` uses Azure Artifact
Signing with GitHub OIDC. It builds on Windows, signs the app, VST3 and CLAP
binaries and generated uninstaller, packages them with NSIS, signs the installer,
verifies signatures and timestamps, and uploads the installer with SHA256SUMS.
It runs manually and does not publish a GitHub release or upload to the website.

Configured signing service:

- Account: `mickibraun`
- Profile: `prelease` (must be Public Trust)
- Region: West US 2
- Endpoint: `https://wus2.codesigning.azure.net/`

## Repository prerequisite

The intended GitHub repository is https://github.com/micki-b/2ndPressing.

This project must be the root of its intended GitHub repository, with `.github/`
at the repository root. At preparation time, the local Git root was the parent
`/Users/michaelbraun/Code`, with no remote configured. Do not publish that parent
directory: it includes unrelated projects. Confirm the intended project repository
before initializing, moving Git metadata, adding a remote, or pushing files.

## Azure connection

1. In Azure, open Microsoft Entra ID > App registrations > New registration.
   Name it `github-secondpressing-signing`, select accounts in this directory
   only, and leave the redirect URI empty. Register it.
2. On its Overview, record Application (client) ID and Directory (tenant) ID.
   Obtain Subscription ID from the subscription containing `mickibraun`.
3. In the app registration, open Certificates & secrets > Federated credentials
   > Add credential. Select the GitHub Actions scenario and supply the exact
   GitHub organization/user `micki-b` (Organization ID `163963902`) and repository
   `2ndPressing` (Repository ID `1460811513`). Choose entity type Environment
   and value `windows-signing`. Name the credential `secondpressing-release`.
   Use the default audience `api://AzureADTokenExchange`.
   The resulting subject must be
   `repo:micki-b@163963902/2ndPressing@1460811513:environment:windows-signing`.
   This repository was created after July 15, 2026 and uses GitHub's immutable
   default subject format, which includes the numeric owner and repository IDs.
4. Open the `prelease` certificate profile's Access control (IAM), or the signing
   account's IAM if profile-level assignment is unavailable. Add the role
   Artifact Signing Certificate Profile Signer to the app's service principal
   `github-secondpressing-signing`. Do not grant it Owner or Identity Verifier.

Reference: https://learn.microsoft.com/en-us/azure/developer/github/connect-from-azure-openid-connect

## GitHub connection

1. In repository Settings > Environments, create `windows-signing`.
2. Configure permitted deployment branches/tags for trusted release code. Where
   supported by the repository plan, configure required reviewers as appropriate.
3. Add these environment secrets:
   - `AZURE_CLIENT_ID`: Application (client) ID from the app registration.
   - `AZURE_TENANT_ID`: Directory (tenant) ID.
   - `AZURE_SUBSCRIPTION_ID`: Azure subscription ID.
4. Commit the workflow and release scripts to the project repository. In Actions,
   choose Windows signed release > Run workflow on the intended release branch.
5. Download `windows-signed-installer` from the completed run. Before publishing,
   test browser download, installation, app audio, plugin discovery, upgrade,
   and uninstall on a clean Windows machine.

No client secret or exported certificate is needed. The federated credential's
repository and environment must match the workflow exactly. An incorrect match
causes Azure login to fail; missing Signer permission or a mismatched service
endpoint can cause signing to fail with HTTP 403.

## NSIS implementation

`GENERATE_UNINSTALLER` makes the helper installer write `stage/Uninstall.exe`
and quit in `.onInit` without installing files. CI signs that uninstaller along
with staged app/plugin binaries. `SIGNED_UNINSTALLER` then embeds the signed
file into the final installer, instead of creating an unsigned one at install
time. The helper is never uploaded as a release artifact.

The existing macOS `build-release.sh` remains an unsigned local build path.
It is not the signed production release workflow. Code signing also does not
guarantee immediate SmartScreen reputation for new releases.

Reference: https://github.com/Azure/artifact-signing-action

## Validation status

The workflow YAML parses and both NSIS modes compile with the existing local
Windows staging files (the final mode used a temporary uninstaller fixture).
NSIS emits warning 6020 in the final mode because it embeds the separately
generated uninstaller instead of calling WriteUninstaller; this is expected.
The Windows PowerShell scripts,
OIDC login, signing calls, and end-to-end installation require the first Windows
Actions run; they cannot be validated on this Mac alone.
