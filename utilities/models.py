from typing import Dict

from transformers import AutoModelForSeq2SeqLM, MBartForConditionalGeneration, MT5ForConditionalGeneration


def get_all_mbart_models() -> Dict[str, MT5ForConditionalGeneration]:
    m1_model = AutoModelForSeq2SeqLM.from_pretrained(
        "/data/n.dallanoce/weights/S2_mbart_pre_en-fr(M1)/checkpoint-180000",
        output_attentions=True)
    m2_model = AutoModelForSeq2SeqLM.from_pretrained(
        "/data/n.dallanoce/weights/S2_mbart_pre_en-fr_de(M2)/checkpoint-180000",
        output_attentions=True)
    m2_model_rply = AutoModelForSeq2SeqLM.from_pretrained(
        "/data/n.dallanoce/weights/S2_mbart_pre_en-fr_de(M2)_replay_8/checkpoint-180000",
        output_attentions=True)
    m3_model = AutoModelForSeq2SeqLM.from_pretrained(
        "/data/n.dallanoce/weights/S2_mbart_pre_en-fr_de_es(M3)/checkpoint-180000",
        output_attentions=True)
    m3_model_rply = AutoModelForSeq2SeqLM.from_pretrained(
        "/data/n.dallanoce/weights/S2_mbart_pre_en-fr_de_es(M3)_replay_8/checkpoint-180000",
        output_attentions=True)
    de_model: MBartForConditionalGeneration = AutoModelForSeq2SeqLM.from_pretrained(
        "/data/n.dallanoce/weights/mbart_pre_en-de/checkpoint-180000", output_attentions=True)

    mf1_2_model_rply: MBartForConditionalGeneration = AutoModelForSeq2SeqLM.from_pretrained(
        "/data/n.dallanoce/weights/mbart_pre_de_ft_en-fr(Mf1-2)_replay_8/checkpoint-90000",
        output_attentions=True)

    mf12_model: MBartForConditionalGeneration = AutoModelForSeq2SeqLM.from_pretrained(
        "/data/n.dallanoce/weights/mbart_pre_de_ft_en-fr(Mf1-2)/checkpoint-100000",
        output_attentions=True)

    mf1_model: MBartForConditionalGeneration = AutoModelForSeq2SeqLM.from_pretrained(
        "/data/n.dallanoce/weights/mbart_ft_en-fr-Mf1_weights_anlsys/checkpoint-100000",
        output_attentions=True)

    mf2_model: MBartForConditionalGeneration = AutoModelForSeq2SeqLM.from_pretrained(
        "/data/n.dallanoce/weights/mbart_ft_en-de-Mf2/checkpoint-100000",
        output_attentions=True)

    mf2_model_rply: MBartForConditionalGeneration = AutoModelForSeq2SeqLM.from_pretrained(
        "/data/n.dallanoce/weights/mbart_ft_en-de-Mf2_replay_8/checkpoint-100000",
        output_attentions=True)

    mf23_model: MBartForConditionalGeneration = AutoModelForSeq2SeqLM.from_pretrained(
        "/data/n.dallanoce/weights/mbart_pre_es_ft_en-de(Mf2-3)/checkpoint-80000",
        output_attentions=True)
    mf23_model_rply: MBartForConditionalGeneration = AutoModelForSeq2SeqLM.from_pretrained(
        "/data/n.dallanoce/weights/mbart_pre_es_ft_en-de(Mf2-3)_replay_8/checkpoint-95000",
        output_attentions=True)
    mf13_model: MBartForConditionalGeneration = AutoModelForSeq2SeqLM.from_pretrained(
        "/data/n.dallanoce/weights/mbart_pre_es_ft_en-fr(Mf3-1)/checkpoint-85000",
        output_attentions=True)
    mf13_model_rply: MBartForConditionalGeneration = AutoModelForSeq2SeqLM.from_pretrained(
        "/data/n.dallanoce/weights/mbart_pre_es_ft_en-fr(Mf3-1)_replay_8/checkpoint-100000",
        output_attentions=True)

    only_ft_en_de: MBartForConditionalGeneration = AutoModelForSeq2SeqLM.from_pretrained(
        "/data/n.dallanoce/weights/mbart_ft_en-de_ft_only/checkpoint-100000",
        output_attentions=True)

    mf3_model: MBartForConditionalGeneration = AutoModelForSeq2SeqLM.from_pretrained(
        "/data/n.dallanoce/weights/mbart_ft_en-es/checkpoint-100000",
        output_attentions=True)

    mf3_model_rply: MBartForConditionalGeneration = AutoModelForSeq2SeqLM.from_pretrained(
        "/data/n.dallanoce/weights/mbart_ft_en-es_replay_8/checkpoint-100000",
        output_attentions=True)

    mf12_model_joint: MBartForConditionalGeneration = AutoModelForSeq2SeqLM.from_pretrained(
        "/data/n.dallanoce/weights/mbart_pre_de_ft_en-fr(Mf1-2)_joint/checkpoint-100000",
        output_attentions=True)
    mf13_model_joint: MBartForConditionalGeneration = AutoModelForSeq2SeqLM.from_pretrained(
        "/data/n.dallanoce/weights/mbart_pre_de_ft_en-fr(Mf1-2)_joint_240k/checkpoint-100000",
        output_attentions=True)
    mf2_model_joint: MBartForConditionalGeneration = AutoModelForSeq2SeqLM.from_pretrained(
        "/data/n.dallanoce/weights/mbart_ft_en-de-Mf2_joint/checkpoint-100000",
        output_attentions=True)
    mf23_model_joint: MBartForConditionalGeneration = AutoModelForSeq2SeqLM.from_pretrained(
        "/data/n.dallanoce/weights/mbart_pre_es_ft_en-de(Mf2-3)_joint_240k/checkpoint-100000",
        output_attentions=True)
    mf3_model_joint: MBartForConditionalGeneration = AutoModelForSeq2SeqLM.from_pretrained(
        "/data/n.dallanoce/weights/mbart_ft_en-es(Mf3)_joint_240k/checkpoint-100000",
        output_attentions=True)
    mbart_str2model = {
        "M1": m1_model,
        "M2": m2_model,
        "M2_de_only": de_model,
        "M2_replay": m2_model_rply,
        "M3": m3_model,
        "M3_replay": m3_model_rply,
        "M1F1": mf1_model,
        "M2F1": mf12_model,
        "M2F1_replay": mf1_2_model_rply,
        "M2F2": mf2_model,
        "MF2_ft_only": only_ft_en_de,
        "M2F2_replay": mf2_model_rply,
        "M3F3": mf3_model,
        "M3F3_replay": mf3_model_rply,
        "M3F2": mf23_model,
        "M3F2_replay": mf23_model_rply,
        "M3F1": mf13_model,
        "M3F1_replay": mf13_model_rply,
        "M2F1_joint": mf12_model_joint,
        "M3F1_joint": mf13_model_joint,
        "M2F2_joint": mf2_model_joint,
        "M3F2_joint": mf23_model_joint,
        "M3F3_joint": mf3_model_joint
    }
    return mbart_str2model


def get_all_mt6_models() -> Dict[str, MBartForConditionalGeneration]:
    mt6_m1 = AutoModelForSeq2SeqLM.from_pretrained(
        "/data/n.dallanoce/weights/mt6_pre_en-fr(M1)_10_10_20_tb/checkpoint-180000",
        output_attentions=True)

    mt6_m2 = AutoModelForSeq2SeqLM.from_pretrained(
        "/data/n.dallanoce/weights/mt6_pre_en-fr_de(M2)_10_20_tb/checkpoint-180000",
        output_attentions=True)
    mt6_m2_rply = AutoModelForSeq2SeqLM.from_pretrained(
        "/data/n.dallanoce/weights/mt6_pre_en-fr_de(M2)_10_20_tb_replay_8/checkpoint-180000",
        output_attentions=True)
    mt6_m3 = AutoModelForSeq2SeqLM.from_pretrained(
        "/data/n.dallanoce/weights/mt6_pre_en-fr_de_es(M3)_10_20_tb/checkpoint-180000",
        output_attentions=True)
    mt6_m3_rply = AutoModelForSeq2SeqLM.from_pretrained(
        "/data/n.dallanoce/weights/mt6_pre_en-fr_de_es(M3)_10_20_tb_replay_8/checkpoint-180000",
        output_attentions=True)

    mt6_mf12_rply = AutoModelForSeq2SeqLM.from_pretrained(
        "/data/n.dallanoce/weights/mt6_pre_de_ft_en-fr(MF1-2)_10_20_tb_replay_8/checkpoint-100000",
        output_attentions=True)

    mt6_mf12 = AutoModelForSeq2SeqLM.from_pretrained(
        "/data/n.dallanoce/weights/mt6_pre_de_ft_en-fr(MF1-2)_10_20_tb/checkpoint-100000",
        output_attentions=True)

    mt6_mf1 = AutoModelForSeq2SeqLM.from_pretrained(
        "/data/n.dallanoce/weights/mt6_ft_en-fr(MF1)_10_10_20_s_tb/checkpoint-100000",
        output_attentions=True)

    mt6_mf2 = AutoModelForSeq2SeqLM.from_pretrained(
        "/data/n.dallanoce/weights/mt6_ft_en-de(MF2)_10_20_tb/checkpoint-100000",
        output_attentions=True)

    mt6_mf2_rply = AutoModelForSeq2SeqLM.from_pretrained(
        "/data/n.dallanoce/weights/mt6_ft_en-de(MF2)_10_20_tb_replay_8/checkpoint-100000",
        output_attentions=True)

    mt6_mf23: AutoModelForSeq2SeqLM = AutoModelForSeq2SeqLM.from_pretrained(
        "/data/n.dallanoce/weights/mt6_pre_es_ft_en-de(MF2-3)_10_10_tb/checkpoint-100000",
        output_attentions=True)

    mt6_mf23_rply = AutoModelForSeq2SeqLM.from_pretrained(
        "/data/n.dallanoce/weights/mt6_pre_es_ft_en-de(MF2-3)_10_20_tb_replay_8/checkpoint-100000",
        output_attentions=True)
    mt6_mf13: AutoModelForSeq2SeqLM = AutoModelForSeq2SeqLM.from_pretrained(
        "/data/n.dallanoce/weights/mt6_pre_es_ft_en-fr-MF3-1_10_20_tb/checkpoint-100000",
        output_attentions=True)
    mt6_mf13_rply = AutoModelForSeq2SeqLM.from_pretrained(
        "/data/n.dallanoce/weights/mt6_pre_es_ft_en-fr-MF3-1_10_20_tb_replay_8/checkpoint-100000",
        output_attentions=True)

    # only_ft_en_de: AutoModelForSeq2SeqLM = AutoModelForSeq2SeqLM.from_pretrained(
    #     "/data/n.dallanoce/weights/mbart_ft_en-de_ft_only/checkpoint-100000",
    #     output_attentions=True)

    mt6_mf3: AutoModelForSeq2SeqLM = AutoModelForSeq2SeqLM.from_pretrained(
        "/data/n.dallanoce/weights/mt6_ft_en-es/checkpoint-100000",
        output_attentions=True)

    mt6_mf3_rply = AutoModelForSeq2SeqLM.from_pretrained(
        "/data/n.dallanoce/weights/mt6_ft_en-es(MF3)_10_20_tb_replay_8/checkpoint-100000",
        output_attentions=True)

    mt6_mf12_joint = AutoModelForSeq2SeqLM.from_pretrained(
        "/data/n.dallanoce/weights/mt6_pre_de_ft_en-fr(MF1-2)_joint/checkpoint-100000",
        output_attentions=True)

    mt6_mf13_joint = AutoModelForSeq2SeqLM.from_pretrained(
        "/data/n.dallanoce/weights/mt6_pre_es_ft_en-fr-MF3-1_joint_240k/checkpoint-30000",
        output_attentions=True)

    mt6_mf23_joint = AutoModelForSeq2SeqLM.from_pretrained(
        "/data/n.dallanoce/weights/mt6_pre_es_ft_en-de(MF2-3)_joint_240k/checkpoint-100000",
        output_attentions=True)

    mt6_mf2_joint = AutoModelForSeq2SeqLM.from_pretrained(
        "/data/n.dallanoce/weights/mt6_ft_en-de(MF2)_joint/checkpoint-100000",
        output_attentions=True)

    mt6_mf3_joint = AutoModelForSeq2SeqLM.from_pretrained(
        "/data/n.dallanoce/weights/mt6_ft_en-es(MF3)_joint_240k/checkpoint-90000",
        output_attentions=True)

    mt6_pre_de_only = AutoModelForSeq2SeqLM.from_pretrained(
        "/data/n.dallanoce/weights/mt6_pre_en-de_20_20_tb/checkpoint-180000",
        output_attentions=True)

    mt6_ft_only_en_de = AutoModelForSeq2SeqLM.from_pretrained(
        "/data/n.dallanoce/weights/mt6_ft_only_en-de_ft_only_tb/checkpoint-90000",
        output_attentions=True)

    mt6_str2model = {
        "M1": mt6_m1,
        "M2": mt6_m2,
        "M2_replay": mt6_m2_rply,
        "M2_de_only": mt6_pre_de_only,
        "M3": mt6_m3,
        "M3_replay": mt6_m3_rply,
        "M1F1": mt6_mf1,
        "M2F1": mt6_mf12,
        "M2F1_replay": mt6_mf12_rply,
        "M2F2": mt6_mf2,
        "M2F2_replay": mt6_mf2_rply,
        "MF2_ft_only": mt6_ft_only_en_de,
        "M3F3": mt6_mf3,
        "M3F3_replay": mt6_mf3_rply,
        "M3F2": mt6_mf23,
        "M3F2_replay": mt6_mf23_rply,
        "M3F1": mt6_mf13,
        "M3F1_replay": mt6_mf13_rply,
        "M2F1_joint": mt6_mf12_joint,
        "M3F1_joint": mt6_mf13_joint,
        "M2F2_joint": mt6_mf2_joint,
        "M3F2_joint": mt6_mf23_joint,
        "M3F3_joint": mt6_mf3_joint
    }
    return mt6_str2model


import gc, torch


class ClearCache:

    def __enter__(self):
        gc.collect()
        torch.cuda.empty_cache()

    def __exit__(self, exc_type, exc_val, exc_tb):
        gc.collect()
        torch.cuda.empty_cache()
