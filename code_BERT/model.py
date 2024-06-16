import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import numpy as np

class CustomDataCollator:
    def __call__(self, features):
        input_ids = torch.stack([f[0] for f in features])
        labels = torch.tensor([f[1] for f in features])

        return {
            'input_ids': input_ids,
            'labels': labels
        }

class TransformerModel:
    def __init__(self, model_name='bert-base-uncased', num_labels=2):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    def tokenize_data(self, texts, max_length=512):
        # Ensure texts are strings
        if isinstance(texts[0], list):
            texts = [" ".join(map(str, text)) for text in texts]
        return self.tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors='pt')

    def train(self, texts, labels, output_dir, epochs=3, batch_size=16):
        print("begin")
        inputs = self.tokenize_data(texts)
        train_inputs, val_inputs, train_labels, val_labels = train_test_split(inputs['input_ids'], labels, test_size=0.1, random_state=42)

        train_dataset = torch.utils.data.TensorDataset(train_inputs, torch.tensor(train_labels))
        val_dataset = torch.utils.data.TensorDataset(val_inputs, torch.tensor(val_labels))
        print("args")
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            evaluation_strategy="epoch",
            save_steps=10_000,
            save_total_limit=2,
            logging_dir='./logs',
            logging_steps=500,
        )
        print("trainer")
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=CustomDataCollator(),  # Utilisation de la fonction de collecte de données personnalisée
            tokenizer=self.tokenizer,
        )
        print("train")
        trainer.train()
        print("###")
        return self.model

    def predict(self, texts, batch_size=16):
        self.model.eval()  # Set the model to evaluation mode
        predictions = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            inputs = self.tokenize_data(batch_texts)
            with torch.no_grad():
                outputs = self.model(**inputs)
            logits = outputs.logits
            batch_predictions = torch.argmax(logits, dim=-1)
            predictions.extend(batch_predictions.cpu().numpy())

        return np.array(predictions)




