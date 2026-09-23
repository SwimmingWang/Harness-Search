# Provenance and third-party notices

This research code builds on [Harness-1](https://github.com/pat-jj/harness-1), including its dataset adapters, tool interfaces, and evaluation infrastructure. The upstream reference revision used for the cookbook dependency is `8ac4012167858f6478fb2a8fd840e4550e2af161`.

The upstream license is preserved verbatim in [licenses/harness-1-LICENSE.txt](licenses/harness-1-LICENSE.txt). This notice does not assign a new license to additional contributions in this repository. Preserve original notices when distributing upstream-derived code.

The runtime imports [Tinker Cookbook](https://github.com/thinking-machines-lab/tinker-cookbook) interfaces from the bundled revision specified in `requirements.txt`. Other Python dependencies remain separately licensed by their authors.

Dataset source links are listed in [docs/data.md](docs/data.md). This repository includes local Web/SEC test query snapshots (questions, answers, and evidence labels), documented in [datasets/README.md](datasets/README.md). Their upstream terms and attribution requirements remain separate from the code license; no new license is assigned to those datasets here. Full retrieval corpora, BrowseComp+, LongSealQA, and model checkpoints are downloaded separately and are not distributed in this repository.
