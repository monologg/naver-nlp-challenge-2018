# -*- coding: utf-8 -*-

import numpy as np
import os
import operator
import pickle
import sys


class Dataset(object):
    def __init__(self, parameter, extern_data):
        self.parameter = parameter
        self.extern_data = extern_data  # 결과적으로 data_loader.py에서 [idx], [ejeol], [ner_tag]를 담은 리스트

        # 만일 necessary.pkl이 존재하지 않을 시, necessary.pkl 생성하는 함수 호출
        # 존재할 시에는 있는 것을 그대로 load함
        if parameter["mode"] == "train" and not os.path.exists(parameter["necessary_file"]):
            self._make_necessary_data_by_train_data()
        else:
            with open(parameter["necessary_file"], 'rb') as f:
                self.necessary_data = pickle.load(f)

        self.parameter["embedding"] = [
            ["word", len(self.necessary_data["word"]), parameter["word_embedding_size"]],  # ["word", word_vocab_len, word_emb_dim]
            ["character", len(self.necessary_data["character"]), parameter["char_embedding_size"]]  # ["character", char_vocab_len, char_emb_dim]
        ]

        self.parameter["n_class"] = len(self.necessary_data["ner_tag"])

    def _make_necessary_data_by_train_data(self):
        # necessary.pkl을 만드는 함수
        necessary_data = {"word": {}, "character": {},
                          "ner_tag": {}, "ner_morph_tag": {}}

        for morphs, tags, ner_tag, ner_mor_list, ner_tag_list in self._read_data_file(extern_data=self.extern_data):
            for mor, tag in zip(morphs, tags):  # 이 tags는 없어도 되는 변수임
                self._check_dictionary(necessary_data["word"], mor)  # word_vocab 만들기

                for char in mor:
                    self._check_dictionary(necessary_data["character"], char)  # char_vocab 만들기

            if type(ner_tag) is list:
                for ne in ner_tag:
                    if ne == "-":
                        continue
                    self._check_dictionary(necessary_data["ner_tag"], ne + "_B")
                    self._check_dictionary(necessary_data["ner_tag"], ne + "_I")
            else:
                self._check_dictionary(necessary_data["ner_tag"], ner_tag + "_B")
                self._check_dictionary(necessary_data["ner_tag"], ner_tag + "_I")

            for nerMor, nerTag in zip(ner_mor_list, ner_tag_list):  # ner_mor_list는 어절, ner_tag_list는 B,I가 포함된 태그
                if nerTag == "-" or nerTag == "-_B":  # TODO: '-' tag인 것은 애초에 담지도 않아서 word vocab 길이의 절반정도임
                    continue
                nerTag = nerTag.split("_")[0]
                self._check_dictionary(necessary_data["ner_morph_tag"], nerMor, nerTag)

        # 존재하는 어절 사전
        # 0: PAD, 1: UNK로 자동으로 세팅해줌
        necessary_data["word"] = self._necessary_data_sorting_and_reverse_dict(necessary_data["word"], start=2)

        # 존재하는 음절 사전
        necessary_data["character"] = self._necessary_data_sorting_and_reverse_dict(necessary_data["character"], start=2)

        # 존재하는 NER 품사 태그 사전
        necessary_data["ner_tag"] = self._necessary_data_sorting_and_reverse_dict(necessary_data["ner_tag"], start=2, unk=False)
        self.ner_tag_size = len(necessary_data["ner_tag"])
        self.necessary_data = necessary_data

        # 존재하는 형태소 별 NER 품사 태그 비율 사전
        # 여기에는 PAD, UNK는 들어가지 않음
        necessary_data["ner_morph_tag"] = self._necessary_data_sorting_and_reverse_dict(necessary_data["ner_morph_tag"], start=0, ner=True)

        with open(self.parameter["necessary_file"], 'wb') as f:
            pickle.dump(necessary_data, f)

    def make_input_data(self, extern_data=None):
        morphs = []
        ne_dicts = []
        characters = []
        labels = []
        sequence_lengths = []
        character_lengths = []
        # sentences = []

        if extern_data is not None:
            self.extern_data = extern_data

        temp = [[], [], []]  # 어절, 어절, BI포함 태그
        # TAG 정보가 없는 경우에는 tag 자리에 mor 정보가 들어온다
        for mor, tag, _, ner_mor, ner_tag in self._read_data_file(pre=False, extern_data=self.extern_data):
            if tag != False:
                temp[0] += mor
                temp[1] += tag
                if len(ner_tag) == 0:
                    temp[2] += ['O'] * len(mor)
                elif len(ner_tag) == len(mor):
                    temp[2] = ner_tag
                else:
                    for i, m in enumerate(mor):
                        if m == ner_mor[0]:
                            break
                    ner_tag = ['O'] * i + ner_tag
                    ner_tag = ner_tag + ['O'] * (len(mor) - len(ner_tag))
                    temp[2] += ner_tag
            else:
                # 단어가 존재하지 않을 시
                morph = [0] * self.parameter["sentence_length"]
                ne_dict = [[0.] * int(self.parameter["n_class"] / 2)] * self.parameter["sentence_length"]  # [15, sent_len]
                character = [[0] * self.parameter["word_length"]] * self.parameter["sentence_length"]
                character_length = [0] * self.parameter["sentence_length"]
                label = [0] * self.parameter["sentence_length"]

                if len(temp[0]) > self.parameter["sentence_length"]:
                    temp = [[], [], []]
                    continue

                sequence_lengths.append(len(temp[0]))
                for mor, tag, neTag, index in zip(temp[0], temp[1], temp[2], range(0, len(temp[0]))):  # mor이나 tag 둘다 어절을 뜻함(tag는 쓰이지 않음)
                    morph[index] = self._search_index_by_dict(self.necessary_data["word"], mor)
                    # ner_morph_tag는 B,I를 무시한 15개의 tag를 반영한 것인데, 애초에 - 태그에 대해서는 포함을 시키지 않은지라
                    # 이 vocab안에 없는 단어(=UNK)일 시에는 [1. 0. 0. ....]의 리스트가 들어감
                    ne_dict[index] = self._search_index_by_dict(self.necessary_data["ner_morph_tag"], mor)
                    if neTag != "-" and neTag != "-_B":
                        label[index] = self._search_index_by_dict(self.necessary_data["ner_tag"], neTag)
                    sub_char = [0] * self.parameter["word_length"]
                    for i, char in enumerate(mor):
                        if i == self.parameter["word_length"]:
                            i -= 1
                            break
                        sub_char[i] = self._search_index_by_dict(self.necessary_data["character"], char)
                    character_length[index] = i + 1
                    character[index] = sub_char

                # aggregate sentences for building bert embedding
                # tokens = temp[0]
                # sentence = " ".join(tokens)
                # sentences.append(sentence)

                morphs.append(morph)
                ne_dicts.append(ne_dict)
                characters.append(character)
                character_lengths.append(character_length)
                labels.append(label)

                temp = [[], [], []]

        self.morphs = np.array(morphs)  # [total_data(90000), max_sent_len]
        self.ne_dicts = np.array(ne_dicts)  # [total_data, max_sent_len, num_classes(15)]
        self.characters = np.array(characters)  # [total_data, max_sent_len, max_char_len]
        self.sequence_lengths = np.array(sequence_lengths)  # [total_data,]
        self.character_lengths = np.array(character_lengths)  # [total_data, max_sent_len]
        self.labels = np.array(labels)  # [total_data, max_sent_len]

    def get_data_batch_size(self, n, train=True):
        if train:
            for i, step in enumerate(range(0, self.parameter["train_lines"], n)):
                if len(self.morphs[step:step + n]) == n:
                    yield self.morphs[step:step + n], self.ne_dicts[step:step + n], self.characters[step:step + n], \
                          self.sequence_lengths[step:step + n], self.character_lengths[step:step + n], \
                          self.labels[step:step + n], i
        else:
            for i, step in enumerate(range(0, self.parameter["train_lines"], n)):
                if len(self.morphs[step:step + n]) == n:
                    yield self.morphs[step:step + n], self.ne_dicts[step:step + n], self.characters[step:step + n], \
                          self.sequence_lengths[step:step + n], self.character_lengths[step:step + n], \
                          self.labels[step:step + n], i

    def _search_index_by_dict(self, dict, key):
        if key in dict:
            return dict[key]
        else:
            if "UNK" in dict:
                return dict["UNK"]
            else:
                temp = [0.0] * int(self.parameter["n_class"] / 2)
                temp[0] = 1.0
                temp = np.array(temp)  # TODO: 원칙상은 이렇게 하는 것이 맞는데, 추후에 확인해볼 것
                return temp

    def _read_data_file(self, pre=True, extern_data=None):
        if extern_data is not None:
            return self._read_extern_data_file(pre, self.extern_data)

    def _read_extern_data_file(self, pre=True, extern_data=None):
        cntLine = 0
        for sentence in extern_data:
            morphs = []
            tags = []
            ner_tag = []
            ner_mor_list = []
            for morph in sentence[1]:  # sentence[1]은 단어(어절)이 모두 담겨있음. 단어들을 looping
                morphs.append(morph)
                tags.append(morph)
                ner_mor_list.append(morph)
            seq_len = len(morphs)

            ner_tag_list = ['O'] * seq_len
            for index, ne in enumerate(sentence[2]):  # sentence[2]에는 ner tag가 담겨있음
                ner_tag.append(ne.split("_")[0])  # B, I tag는 무시하고 담음
                ner_tag_list[index] = ne  # B, I tag가 포함된 것을 그대로 담음
            '''
            morphs : 어절
            tags : 어절 (사실 이건 없어도 될 듯)
            ner_tag : B,I를 제외한 총 15개의 tag (O 포함)
            ner_mor_list : 어절
            ner_tag_list : B,I를 포함한 총 29개의 tag
            '''
            yield morphs, tags, ner_tag, ner_mor_list, ner_tag_list
            # yield 호출 후 그 다음 라인들 진행 (cnt 변수 1 증가 시켜주기)
            cntLine += 1
            if pre == False:
                yield [], False, False, False, False
            if cntLine % 1000 == 0:
                sys.stderr.write("%d Lines .... \r" % (cntLine))

                if self.parameter["train_lines"] < cntLine:
                    break

    def _check_dictionary(self, dict, data, value=0):
        if type(value) is int:
            if not data in dict:
                dict[data] = value
        elif type(value) is str:
            if not value in dict:
                dict[data] = {value: 1}
            else:
                if value in dict[data]:
                    dict[data][value] += 1
                else:
                    dict[data][value] = 1

    def _necessary_data_sorting_and_reverse_dict(self, dict, start=1, unk=True, ner=False):
        dict_temp = {}
        index = start

        if start == 2:
            dict_temp["PAD"] = 0
            if unk:
                dict_temp["UNK"] = 1
            else:
                dict_temp["O"] = 1
        elif start == 1:
            dict_temp["PAD"] = 0
        elif start == 3:
            dict_temp["PAD"] = 0
            dict_temp["UNK"] = 1
            dict_temp["GO"] = 2

        for key in sorted(dict.items(), key=operator.itemgetter(0), reverse=False):
            if ner:
                items = np.zeros(int(self.ner_tag_size / 2))
                for i in key[1]:
                    items[int(self.necessary_data["ner_tag"][i + "_B"] / 2)] = dict[key[0]][i]
                dict_temp[key[0]] = items / np.sum(items)
            else:
                dict_temp[key[0]] = index
                index += 1

        return dict_temp
