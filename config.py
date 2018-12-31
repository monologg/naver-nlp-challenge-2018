# -*- coding: utf-8 -*-


class Config(object):
    # 반드시 해당 코드에 옵션을 추가할때  __init__과 parse_arg 함수 둘 다 추가해줘야 한다!
    def __init__(self):
        # 0. Dev size
        self.dev_size = 4000
        self.patience = 3
        self.lr_decay = 0.1

        # 1. Basic Train Setting
        self.nb_epochs = 6
        self.batch_size = 16
        self.lr_rate = 0.001
        self.keep_prob = 0.75
        self.lstm_keep_prob = 0.9

        # 2. Word Embedding
        self.word_embedding_size = 300
        self.char_embedding_size = 300
        # 둘다 trainable로 하는 것이 성능이 더 좋은 듯
        self.use_word_pretrained = True
        self.use_char_pretrained = True
        self.word_embedding_trainable = True
        self.char_embedding_trainable = True

        # 3. LSTM, length setting
        self.lstm_units = 256
        self.char_lstm_units = 128
        self.max_sent_len = 180
        self.max_word_len = 8

        # 4. Attention
        self.use_self_attention = True
        self.num_heads = 8
        assert self.lstm_units % self.num_heads == 0
        self.penalize_self_align = True

        # 5. Regularization
        self.use_reg_loss = True
        self.reg_lambda = 1e-7

        # 6. Exponential moving Average
        self.use_ema = True
        self.ema_decay_rate = 0.999

        # 7.1. lstm weight dropout
        self.use_custom_lstm_cell = False
        self.weight_keep_prob = 0.75

        # 7.2. LSTM Variational Dropout
        self.use_variational_dropout = True

        # 8. Convolution
        self.use_conv_model = True
        self.kernel_sizes = [2, 3, 4]
        self.num_filters = 64
        self.use_positional_embedding = True

        # 9. Highway Network
        self.use_highway = True
        self.num_layers = 2

        # 10. Language model
        self.use_lm = True
        self.gamma = 0.5  # importance of language model loss

        # 11. Add dropout after embedding
        self.use_dropout_after_embedding = False
        self.emb_keep_prob = 0.9

        # 12. Use addition dense layer
        self.use_additional_dense = False
        self.dense_unit_size = 256
        self.dense_keep_prob = 0.9
        self.use_non_linear_on_dense = True

        # 13. gradient clip
        self.use_grad_clip = False
        self.clip = 5
        # ------------------------------------------- #
        if self.use_word_pretrained:
            self.word_embedding_size = 300
        if self.use_char_pretrained:
            self.char_embedding_size = 300
        assert self.clip < 5.1

    def parse_arg(self, parser):
        '''
        :param parser: 외부에서 정의된 parser객체를 받음 
        :return: parser 객체 리턴
        '''
        parser.add_argument('--verbose', default=False, required=False, action='store_true', help='verbose')

        parser.add_argument('--mode', type=str, default="train", required=False, help='Choice operation mode')
        parser.add_argument('--iteration', type=int, default=0, help='fork 명령어를 사용할때 iteration 값에 매칭되는 모델이 로드됩니다.')
        parser.add_argument('--pause', type=int, default=0, help='모델이 load 될때 1로 설정됩니다.')

        parser.add_argument('--input_dir', type=str, default="data_in", required=False, help='Input data directory')
        parser.add_argument('--output_dir', type=str, default="data_out", required=False, help='Output data directory')
        parser.add_argument('--necessary_file', type=str, default="necessary.pkl")
        parser.add_argument('--train_lines', type=int, default=100000, required=False, help='Maximum train lines')

        parser.add_argument('--epochs', type=int, default=self.nb_epochs, required=False, help='Epoch value')
        parser.add_argument('--batch_size', type=int, default=self.batch_size, required=False, help='Batch size')
        parser.add_argument('--learning_rate', type=float, default=self.lr_rate, required=False, help='Set learning rate')
        parser.add_argument('--keep_prob', type=float, default=self.keep_prob, required=False, help='Dropout_rate')
        parser.add_argument("--lstm_keep_prob", type=float, default=self.lstm_keep_prob, required=False,
                            help="lstm input, output dropout rate")  # Variational이 아니어도 해당됨

        # WARN: 만일 pretrained을 사용할 시, word_embedding_size나 char_embedding_size를 300으로 반드시 바꿔야 함
        parser.add_argument('--use_word_pretrained', type=bool, default=self.use_word_pretrained, required=False,
                            help='Using word pretrained embedding')
        parser.add_argument('--use_char_pretrained', type=bool, default=self.use_char_pretrained, required=False,
                            help='Using char pretrained embedding')

        parser.add_argument('--word_embedding_trainble', type=bool, default=self.word_embedding_trainable, required=False,
                            help='Word embedding trainable or not')
        parser.add_argument('--char_embedding_trainble', type=bool, default=self.char_embedding_trainable, required=False,
                            help='Character embedding trainable or not')

        parser.add_argument("--word_embedding_size", type=int, default=self.word_embedding_size, required=False, help='Word, WordPos Embedding Size')
        parser.add_argument("--char_embedding_size", type=int, default=self.char_embedding_size, required=False, help='Char Embedding Size')
        parser.add_argument("--tag_embedding_size", type=int, default=16, required=False, help='Tag Embedding Size')

        parser.add_argument('--lstm_units', type=int, default=self.lstm_units, required=False, help='Hidden unit size')
        parser.add_argument('--char_lstm_units', type=int, default=self.char_lstm_units, required=False, help='Hidden unit size for Char rnn')
        parser.add_argument('--sentence_length', type=int, default=self.max_sent_len, required=False, help='Maximum words in sentence')
        parser.add_argument('--word_length', type=int, default=self.max_word_len, required=False, help='Maximum chars in word')

        parser.add_argument("--use_self_attention", type=bool, default=self.use_self_attention, required=False, help="use_self_attention")
        parser.add_argument('--num_heads', type=int, default=self.num_heads, required=False, help='num heads')  # Self Attention Setting
        parser.add_argument("--penalize_self_align", type=bool, default=self.penalize_self_align, required=False, help="penalize self alignment")

        parser.add_argument("--use_reg_loss", type=bool, default=self.use_reg_loss, required=False, help="l2 regularization loss")  # Regularization
        parser.add_argument("--lambda", type=float, default=self.reg_lambda, required=False, help="l2 regularization lambda")  # Regularization

        # Exponential Moving Average 사용 여부
        parser.add_argument("--use_ema", type=bool, default=self.use_ema, required=False, help="Use exponential moving average")
        parser.add_argument("--ema_decay_rate", type=float, default=self.ema_decay_rate, required=False, help="exponential moving average decay rate")

        # Customized LSTM Cell 사용 여부
        parser.add_argument("--use_custom_lstm_cell", type=bool, default=self.use_custom_lstm_cell, required=False, help="Use custom lstm cell")
        parser.add_argument("--weight_keep_prob", type=float, default=self.weight_keep_prob, required=False,
                            help="weight dropout for custom lstm cell")

        # LSTM에 Variational Dropout 사용 여부 체크
        parser.add_argument("--use_variational_dropout", type=bool, default=self.use_variational_dropout, required=False,
                            help="Use variational dropout")

        # ConvModel 사용 여부 체크
        parser.add_argument("--use_conv_model", type=bool, default=self.use_conv_model, required=False, help="Use Conv Model")
        parser.add_argument("--kernel_sizes", type=list, default=self.kernel_sizes, required=False, help="kernel sizes for conv")
        parser.add_argument("--num_filters", type=int, default=self.num_filters, required=False, help="num filters for conv")
        parser.add_argument("--use_positional_embedding", type=bool, default=self.use_positional_embedding, required=False,
                            help="Use positional embedding")

        # highway network 사용 여부
        parser.add_argument("--use_highway", type=bool, default=self.use_highway, required=False, help="use highway network")
        parser.add_argument("--num_layers", type=int, default=self.num_layers, required=False, help="the number of layers for highway network")

        # language model 사용 여부
        parser.add_argument("--use_lm", type=bool, default=self.use_lm, required=False, help="use language model")
        parser.add_argument("--gamma", type=float, default=self.gamma, required=False, help="importance of language mode loss")

        # embedding 다음에 dropout 사용
        parser.add_argument("--use_dropout_after_embedding", type=bool, default=self.use_dropout_after_embedding, required=False,
                            help="Use_dropout_after_embedding")
        parser.add_argument("--emb_keep_prob", type=float, default=self.emb_keep_prob, required=False, help="emb_keep_prob")

        # 추가적인 Dense layer
        parser.add_argument("--use_additional_dense", type=bool, default=self.use_additional_dense, required=False, help="Use additional dense layer")
        parser.add_argument("--dense_unit_size", type=int, default=self.dense_unit_size, required=False, help="Dense layer unit size on middle")
        parser.add_argument("--dense_keep_prob", type=float, default=self.dense_keep_prob, required=False, help="dense_keep_prob")
        parser.add_argument("--use_non_linear_on_dense", type=bool, default=self.use_non_linear_on_dense, required=False,
                            help="Use non linear activation on Dense layer")

        # grad clip
        parser.add_argument("--use_grad_clip", type=bool, default=self.use_grad_clip, required=False, help="Use grad clip")
        parser.add_argument("--clip", type=float, default=self.clip, required=False, help="clip")

        parser.add_argument("--lr_decay", type=float, default=self.lr_decay, required=False, help="lr_decay")

        return parser
