# Contributing

Thank you for your interest in 2ndPressing.

## Reports and ideas: very welcome

Please use the issue forms:

- **Bug report** for crashes, no sound, installer, activation or website problems
- **Sound quality** for clicks left in, music damaged, percussion dulled or other
  artifacts. These are the most useful reports; a short clip helps most.
- **Feature request** for ideas

Security problems go through private reporting instead; see
[SECURITY.md](SECURITY.md).

## Code: by arrangement, for now

2ndPressing is GPL-3.0 and developed by one person. Code pull requests aren't
accepted by default yet: the author holds the copyright to all of the code,
which keeps options open for distribution channels whose terms conflict with
the GPL (such as the Mac App Store). Accepting outside code under the GPL
would close those options.

If you'd like to contribute code, open an issue describing the change first
and we'll agree how (it may involve a short contributor agreement). Forks are
of course welcome under the GPL.

## Development setup

See [README.md](README.md) and [runtime/README.md](runtime/README.md). Tests:

```sh
cd runtime && cargo test --workspace     # engine, plugin, app, CLI
cd tools && uv run pytest                # data, training and evaluation tools
```

Pull requests run the same tests on GitHub (Tests workflow).
