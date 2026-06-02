from torch import nn
import torch
from typing import Type
from torch.utils.data import DataLoader
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select, func
from collections import defaultdict

from ..database.models import InatClassificationDataset
from ..models.loaders.c_dataloader import class_dataloader
from ..models.c_custom_resnet import CustomResnet, BasicBlock
from ..database.config import get_session_factory

class CustomResnetTrainer:

    def __init__(self,
                 model: Type[nn.Module],
                 dataloader: DataLoader = None,
                 session_maker: sessionmaker = get_session_factory()
                 ):

        self.session = session_maker()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        col_set = {"species": InatClassificationDataset.zero_indexed_species,
                   "genus": InatClassificationDataset.zero_indexed_genus,
                   "subfamily": InatClassificationDataset.zero_indexed_subfamily}

        with self.session as session:
            rows = session.execute(
                select(
                    func.max(InatClassificationDataset.zero_indexed_species),
                    func.max(InatClassificationDataset.zero_indexed_genus),
                    func.max(InatClassificationDataset.zero_indexed_subfamily)
                )
            )
            self.num_labels = [[r[0], r[1], r[2]] for r in rows][0]

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

        self.dataloader = dataloader if dataloader else class_dataloader()
        self.model = model if model else CustomResnet(
            block=BasicBlock,
            layers=[8, 8, 12, 6],
            num_class=self.num_labels
        )
        self.criterion_species = nn.CrossEntropyLoss(weight=torch.tensor(weight_dict["species"]).to(self.device))
        self.criterion_genus = nn.CrossEntropyLoss(weight=torch.tensor(weight_dict["genus"]).to(self.device))
        self.criterion_subfamily = nn.CrossEntropyLoss(weight=torch.tensor(weight_dict["subfamily"]).to(self.device))

    def train_one_epoch(self):
        #Test comment
        raise NotImplementedError
















