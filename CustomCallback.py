import transformers
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl


class CustomCallback(TrainerCallback):
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        model = kwargs['model']
        loss = model.loss.item()
        print(f"train loss {loss}")
