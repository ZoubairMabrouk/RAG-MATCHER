from .bert import BERTStrategy


class ClinicalBERTStrategy(BERTStrategy):

    DEFAULT_MODEL = (
        "emilyalsentzer/Bio_ClinicalBERT"
    )

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str | None = None,
    ):
        super().__init__(
            model_name=model_name,
            device=device,
        )