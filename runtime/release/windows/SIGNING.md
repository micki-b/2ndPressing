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

## Repository

The workflow lives in https://github.com/micki-b/2ndPressing, with `.github/`
at the repository root. The account name, endpoint and repository IDs in this
file are not secrets: Azure only issues signing tokens to workflow jobs whose
OIDC subject matches the federated credential below exactly.

## Azure connection

1. In Azure, open Microsoft Entra ID > App registrations > New registration.
   Name it `github-secondpressing-signing`, select accounts in this directory
   only, and leave the redirect URI empty. Register it.
2. On its Overview, record Application (client) ID and Directory (tenant) ID.
   Obtain Subscription ID from the subscription containing `mickibraun`.
3. In the app registration, open Certificates & secrets > Federated credentials
   > Add credential. Select the GitHub Actions scenario and supply the exact
   GitHub organization/user `micki-b` (Organization ID `163963902`) and repository
   `2ndPressing` (Repository ID `1466029706`). Choose entity type Environment
   and value `windows-signing`. Name the credential `secondpressing-release`.
   Use the default audience `api://AzureADTokenExchange`.
   The resulting subject must be
   `repo:micki-b@163963902/2ndPressing@1466029706:environment:windows-signing`.
   This repository was created after July 15, 2026 and uses GitHub's immutable
   default subject format, which includes the numeric owner and repository IDs.
4. Open the `prelease` certificate profile's Access control (IAM), or the signing
   account's IAM if profile-level assignment is unavailable. Add the role
   Artifact Signing Certificate Profile Signer to the app's service principal
   `github-secondpressing-signing`. Do not grant it Owner or Identity Verifier.

Reference: https://learn.microsoft.com/en-us/azure/developer/github/connect-from-azure-openid-connect

## GitHub connection

1. In repository Settings > Environments, create `windows-signing`.
2. Protect it. This environment is what Azure trusts, so whatever can run a job
   in it can sign as the publisher:
   - Deployment branches and tags: selected branches, `main` only.
   - Required reviewers: the maintainer, so every signing run waits for approval.
   - Allow administrators to bypass: off.
   All three are set on the public repository.
3. Add these environment secrets:
   - `AZURE_CLIENT_ID`: Application (client) ID from the app registration.
   - `AZURE_TENANT_ID`: Directory (tenant) ID.
   - `AZURE_SUBSCRIPTION_ID`: Azure subscription ID.
4. Commit the workflow and release scripts to the project repository. In Actions,
   choose Windows signed release > Run workflow on the intended release branch.
5. Download `windows-signed-installer` from the completed run. Before publishing,
   test browser download, installation, app audio, plugin discovery, upgrade,
   and uninstall on a clean Windows machine.

For packaging/signing retries, the optional `unsigned_run_id` input restores
`windows-unsigned-staging` from an earlier run in this repository. The workflow
first checks that run was this workflow, started by hand, on `main`, so a build
from another branch can't be signed this way. These diagnostic artifacts expire
after one day. Leave the input empty for normal releases so the workflow builds
the selected source revision.

## Supply chain

Every action in the workflow is pinned to a full commit SHA, with the version
in a comment, and NSIS is pinned to 3.10.0. Dependabot (`.github/dependabot.yml`)
proposes action updates as pull requests; review what changed before merging,
since these steps run with signing access.

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

Windows compilation passed in run 44009622394. Run 44011406548 reused those
compiled binaries and successfully generated the uninstaller, authenticated via
OIDC, signed all four staged binaries, packaged and signed the installer,
verified signatures and timestamps, and uploaded the release artifact.

https://github.com/micki-b/2ndPressing/actions/runs/44011406548

The downloaded installer checksum was also checked on the Mac. (That run wrote
SHA256SUMS.txt with a Windows line ending, which `shasum -c` misreads; the
workflow now writes a plain line feed.) Installation,
audio playback, plugin discovery, upgrades, and uninstall still need a clean
Windows machine test. NSIS warning 6020 in the final packaging mode is expected:
that mode embeds the separately generated, signed uninstaller.
