from dotenv import load_dotenv
from sqlalchemy import select, func
from ..database.config import get_session_factory
from ..database.models import InatClassificationDataset, InatTaxa
from collections import defaultdict
from PIL import Image
from pathlib import Path
from ..transforms.letterbox_resize import LetterboxResize
import numpy as np


def run():
    # session = get_session_factory()()
    # letterbox = LetterboxResize(320)
    # rows = session.execute(
    #     select(
    #         InatClassificationDataset.taxon_id,
    #         InatClassificationDataset.filename,
    #         InatClassificationDataset.proposed_bbox,
    #     ).where(InatClassificationDataset.proposed_bbox != "null")
    # )
    #
    # dictrows = [{"filename": r.filename, "bbox": r.proposed_bbox} for r in rows][2]
    # print(dictrows["filename"], dictrows["bbox"])
    # bbox = dictrows["bbox"]
    # filename = dictrows["filename"]
    # x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
    # x1 -= x1 * 0.15 / 2
    # y1 -= y1 * 0.15 / 2
    # x2 += x2 * 0.15 / 2
    # y2 += y2 * 0.15 / 2
    #
    # print(x1, x2, y1, y2)
    #
    # with Image.open(
    #     Path(__file__).parents[1] / "data/classification_images" / filename
    # ) as img:
    #     print(img.size[0], img.size[1])
    #     img2 = img.crop((x1, x2, y1, y2))
    #     img2.save("./cropped_image.jpg")
    #     img3 = letterbox(img2)
    #     img3.save("./letterbox_resize.jpg")

    def _build_stmt(label: str):
        return select(InatTaxa.taxon_id.label(label)).where(InatTaxa.rank == label)

    ancestry_dict = {}
    session_factory = get_session_factory()
    with session_factory() as session:
        for label in ["family", "genus"]:
            rows = session.execute(_build_stmt(label)).all()
            ancestry_dict[label] = [row[0] for row in rows]

    print(ancestry_dict["family"][:5])
    print(ancestry_dict["genus"][:5])


if __name__ == "__main__":
    load_dotenv()

    run()
