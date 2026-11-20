# Security policy

## Reporting a vulnerability

Please report security problems privately, not in a public issue:

- On GitHub: **Security › Report a vulnerability** in this repository, or
- by email: hello@secondpressing.com

Include what you found, how to reproduce it, and which version or page is
affected. You'll get a reply within a week. This is a one-person, unpaid
project, so there is no bug bounty, but reports are credited if you'd like.

## Scope

- The 2ndPressing app, plugins (Audio Unit, VST3, CLAP) and command-line tool
- The installers and their code signatures
- The website, including the activation-code registration at
  https://secondpressing.com/register/

## Notes

- Activation codes are checked offline and the software never contacts the
  internet. A way to make 2ndPressing run without a code is not a
  vulnerability: the source is GPL, and the code exists for registration, not
  copy protection.
- Releases are signed: macOS with an Apple Developer ID (notarized), Windows
  through Azure Artifact Signing (publisher "Michael Braun"). Please report any
  2ndPressing build that isn't signed by these.
