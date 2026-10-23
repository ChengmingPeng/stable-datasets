import io
import zipfile

from PIL import Image

from stable_datasets.images import CitrusLeaves, TrashNet


def make_builder(builder_class):
    """Initialize a builder without downloading the dataset."""
    builder = object.__new__(builder_class)
    builder_class.__init__(builder)
    return builder


def make_image(image_format):
    buffer = io.BytesIO()
    image = Image.new("RGB", (16, 16), color="green")
    image.save(buffer, format=image_format)
    return buffer.getvalue()


def test_citrus_leaves(tmp_path):
    builder = make_builder(CitrusLeaves)
    classes = builder.info.features["label"].names

    assert len(classes) == 4
    assert builder.info.supervised_keys == ("image", "label")

    archive_path = tmp_path / "citrus.zip"

    with zipfile.ZipFile(archive_path, "w") as archive:
        for label in classes:
            archive.writestr(
                f"Citrus/Leaves/{label}/example.png",
                make_image("PNG"),
            )

    examples = list(builder._generate_examples(archive_path, "train"))

    assert len(examples) == 4
    assert {row["label"] for _, row in examples} == set(range(4))

    for _, row in examples:
        with Image.open(io.BytesIO(row["image"])) as image:
            image.verify()


def test_trashnet(tmp_path):
    builder = make_builder(TrashNet)
    classes = builder.info.features["label"].names

    assert len(classes) == 6
    assert builder.info.supervised_keys == ("image", "label")

    archive_path = tmp_path / "trashnet.zip"

    with zipfile.ZipFile(archive_path, "w") as archive:
        for label in classes:
            for index in range(5):
                archive.writestr(
                    f"dataset-resized/{label}/{index}.jpg",
                    make_image("JPEG"),
                )

        # Ignore macOS metadata files.
        archive.writestr(
            "__MACOSX/dataset-resized/cardboard/._cardboard9.jpg",
            b"not a real image",
        )

    train = list(builder._generate_examples(archive_path, "train"))
    test = list(builder._generate_examples(archive_path, "test"))

    assert len(train) == 24
    assert len(test) == 6

    train_keys = {key for key, _ in train}
    test_keys = {key for key, _ in test}

    assert train_keys.isdisjoint(test_keys)
    assert len(train_keys | test_keys) == 30

    assert {row["label"] for _, row in train} == set(range(6))
    assert {row["label"] for _, row in test} == set(range(6))

    for _, row in train + test:
        with Image.open(io.BytesIO(row["image"])) as image:
            image.verify()
