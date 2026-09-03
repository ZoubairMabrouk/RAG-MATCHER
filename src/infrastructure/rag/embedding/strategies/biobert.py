from .bert import BERTStrategy


class BioBERTStrategy(BERTStrategy):

    DEFAULT_MODEL = (
        "dmis-lab/biobert-base-cased-v1.2"
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