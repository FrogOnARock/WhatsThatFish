from dotenv import load_dotenv
from sqlalchemy import select, func
import torch
import numpy as np
from ..database.config import get_session_factory
from ..database.models import InatClassificationDataset
from collections import defaultdict

def run():
    session = get_session_factory()()
    col_set = {"species": InatClassificationDataset.zero_indexed_species,
               "genus": InatClassificationDataset.zero_indexed_genus,
               "subfamily": InatClassificationDataset.zero_indexed_subfamily}

    with session:
        rows = session.execute(
            select(
                func.max(InatClassificationDataset.zero_indexed_species),
                func.max(InatClassificationDataset.zero_indexed_genus),
                func.max(InatClassificationDataset.zero_indexed_subfamily)
            )
        )
        num_labels = [[r[0], r[1], r[2]] for r in rows][0]

        weight_dict = defaultdict(list)
        for lbl, col in col_set.items():
            rows = session.execute(
                select(
                    col,
                    func.round(
                        (func.sum(func.count()).over() / func.count(col).over()) / func.count(),
                        4).label("weight")
                ).group_by(col)
                .order_by(col.desc())
            )
            weight_dict[lbl] = [float(r.weight) for r in rows]

    print(num_labels)
    print(weight_dict["species"][:10])
    print(torch.tensor(weight_dict["species"]))

if __name__ == '__main__':
    load_dotenv()

    run()