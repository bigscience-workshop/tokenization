import json
from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--tokenizer-json-path", type=str, help="Path to `tokenizer.json` file")
    return parser.parse_args()

def _remove_replace(data):
    normalizer = data["normalizer"]
    if normalizer["type"] == "Sequence":
        normalizers = normalizer["normalizers"]
        assert len(normalizers) == 2

        updated_normalizers = [elt for elt in normalizers if elt["type"] != "Replace"]

        assert len(updated_normalizers) == 1

        data["normalizer"] = updated_normalizers[0]
        normalizer = data["normalizer"]

    assert normalizer["type"] == "Precompiled"
    return data

def _add_empty_strings(data):
    return data


def main():
    args = get_args()

    with open(args.tokenizer_json_path, "r") as fi:
        data = json.load(fi)

    data = _remove_replace(data)
    data = _add_empty_strings(data)

    with open(args.tokenizer_json_path, "w") as fo:
        json.dump(data, fo, indent=2)


if __name__ == "__main__":
    main()