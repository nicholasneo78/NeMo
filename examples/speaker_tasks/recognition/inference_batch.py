import os
import json
from tqdm import tqdm
from typing import List, Dict
import torch
import soundfile as sf
from nemo.collections.asr.models import EncDecSpeakerLabelModel

from sklearn.metrics import classification_report, confusion_matrix

class LIDInference:

    def __init__(
        self,
        model_dir: str,
        root_dir: str,
        input_manifest_dir: str,
        output_manifest_dir: str,
        inference_batch: str=32,
    ) -> None:
        
        self.root_dir = root_dir
        self.model_dir = model_dir
        self.input_manifest_dir = input_manifest_dir
        self.output_manifest_dir = output_manifest_dir
        self.inference_batch = inference_batch
        
        self.model = EncDecSpeakerLabelModel.restore_from(restore_path=self.model_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        

    def load_nemo_manifest(self) -> List[Dict[str, str]]:

        with open(self.input_manifest_dir, 'r+', encoding='utf-8') as fr:
            lines = fr.readlines()
            items = [json.loads(line.strip('\r\n')) for line in lines]

        return items


    def load_audio(self, path) -> torch.tensor:
        signal, sr = sf.read(path)
        if sr != 16000:
            raise ValueError(f"{path} - Sample rate must be 16 kHz, got {sr}")
        return torch.tensor(signal, dtype=torch.float32)
    

    def collate_batch(self, batch_signals):

        max_len = max(sig.shape[0] for sig in batch_signals)
        padded_signals = [torch.nn.functional.pad(sig, (0, max_len - sig.shape[0])) for sig in batch_signals]
        stacked = torch.stack(padded_signals)
        lengths = torch.tensor([sig.shape[0] for sig in batch_signals], dtype=torch.int64)
        return stacked.to(self.device), lengths.to(self.device)


    def convert_confidence_score_list_to_dict(self, language_list: List[str], confidence_dict_list: List[str]) -> Dict[str, float]:

        """
        this is specifically for data cartography, converting the confidence list into dictionary with language tag in it

        eg. [0.4447, 0.28994, 0.26535] -> {"en": 0.4447, "id": 0.28994, "ms": 0.26535}
        """

        if len(language_list) != len(confidence_dict_list):
            raise ValueError("The length of 2 list does not match!")

        # unbatch and iterate the confidence values into single list
        mapping_dict = {}

        for lang, conf in zip(language_list, confidence_dict_list):
            mapping_dict[lang] = round(conf, 8)

        return mapping_dict


    def infer(self) -> None:

        output_list = []
        items = self.load_nemo_manifest()
        label_list = self.model.cfg.train_ds.labels

        for item in tqdm(range(0, len(items), self.inference_batch)):
            batch_items = items[item:item+self.inference_batch]
            batch_paths = [os.path.join(self.root_dir, item["audio_filepath"]) for item in batch_items]
            signals = [self.load_audio(path) for path in batch_paths]
            input_signal, input_signal_length = self.collate_batch(signals)

            with torch.no_grad():
                logits, _ = self.model(input_signal=input_signal, input_signal_length=input_signal_length)
                probs = torch.softmax(logits, dim=-1)
                predicted_index = torch.argmax(probs, dim=-1)
                # confidences = torch.max(probs, dim=-1).values
                batch_confidences = [
                    self.convert_confidence_score_list_to_dict(
                        language_list=label_list,
                        confidence_dict_list=sample.tolist()
                    )
                    for sample in probs
                ]

            for batch_item, index, batch_conf in zip(batch_items, predicted_index.tolist(), batch_confidences):
                label = label_list[index]
                batch_item['pred_language'] = label
                batch_item['confidence'] = batch_conf[label]
                batch_item['confidence_full'] = batch_conf

                output_list.append(batch_item)

        # export manifest
        with open(self.output_manifest_dir, 'w+', encoding='utf-8') as fw:
            for item in output_list:
                fw.write(json.dumps(item) + '\n')


    def __call__(self) -> None:

        return self.infer()
    

class LIDEvaluation:

    def __init__(self, model_dir: str, manifest_dir: str) -> None:

        self.model_dir = model_dir
        self.manifest_dir = manifest_dir
        self.model = EncDecSpeakerLabelModel.restore_from(restore_path=self.model_dir)
        self.label_list = self.model.cfg.train_ds.labels

    def load_nemo_manifest(self):

        with open(self.manifest_dir, 'r+', encoding='utf-8') as fr:
            lines = fr.readlines()
            items = [json.loads(line.strip('\r\n')) for line in lines]

        return items


    def compute_metrics(self) -> None:
        
        items = self.load_nemo_manifest()

        ref_list = [item['language'] for item in items]
        pred_list = [item['pred_language'] for item in items]

        print(f"Reference label order for the confusion matrix: {self.label_list}")
        print(confusion_matrix(y_true=ref_list, y_pred=pred_list, labels=self.label_list))
        print(classification_report(y_true=ref_list, y_pred=pred_list, labels=self.label_list, digits=4))


    def __call__(self) -> None:

        return self.compute_metrics()


if __name__ == "__main__":

    MODEL_DIR = "/models/nemo_test/TitaNet-Finetune/2025-05-08_09-21-18-ReduceLROnPlateau-no-ES/checkpoints/TitaNet-Finetune.nemo"
    ROOT_DIR = "/datasets/mms_lid/test_split"
    INPUT_MANIFEST_DIR = os.path.join(ROOT_DIR, "test_manifest.json")
    OUTPUT_MANIFEST_DIR = os.path.join(ROOT_DIR, "pred_test_manifest_nemo.json")
    INFERENCE_BATCH=32
    
    RUN_INFERENCE=False
    HAS_LABEL=True

    if RUN_INFERENCE:
        l = LIDInference(
            model_dir=MODEL_DIR,
            root_dir=ROOT_DIR,
            input_manifest_dir=INPUT_MANIFEST_DIR,
            output_manifest_dir=OUTPUT_MANIFEST_DIR,
            inference_batch=INFERENCE_BATCH
        )()

    if HAS_LABEL:
        ev = LIDEvaluation(
            model_dir=MODEL_DIR,
            manifest_dir=OUTPUT_MANIFEST_DIR,
        )()