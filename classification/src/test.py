# from datasets import load_dataset
# from transformers import AutoFeatureExtractor, HubertForSequenceClassification
#
# dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
# dataset = dataset.sort("id")
# sampling_rate = dataset.features["audio"].sampling_rate
#
# feature_extractor = AutoFeatureExtractor.from_pretrained("superb/hubert-base-superb-ks")
# model = HubertForSequenceClassification.from_pretrained("superb/hubert-base-superb-ks")
# x = [dataset[i]["audio"]["array"][:1000] for i in range(0, 8)]
# inputs = feature_extractor(x, return_tensors="pt")
# logits = model(**inputs).logits
# print(logits)

#
# from transformers import SpeechT5Processor, SpeechT5ForSpeechToSpeech, SpeechT5HifiGan, set_seed
# from datasets import load_dataset
# import torch
#
# dataset = load_dataset(
#     "hf-internal-testing/librispeech_asr_demo", "clean", split="validation"
# )  # doctest: +IGNORE_RESULT
# dataset = dataset.sort("id")
# sampling_rate = dataset.features["audio"].sampling_rate
#
# processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_vc")
# model = SpeechT5ForSpeechToSpeech.from_pretrained("microsoft/speecht5_vc")
# vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
#
# # audio file is decoded on the fly
# inputs = processor(audio=dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")
#
# speaker_embeddings = torch.zeros((1, 512))  # or load xvectors from a file
#
# set_seed(555)  # make deterministic
#
# # generate speech
# speech = model.generate_speech(inputs["input_values"], speaker_embeddings, vocoder=vocoder)
# print(speech)

















