from pickletools import optimize
from torch import nn
from torch.functional import F
from constants import *
from layers import *
from data import pad_or_truncate
from schedulers import WarumUpInverseSquareRootScheduler
from util import *

import torchmetrics
import torch
import heapq
import math
import pytorch_lightning as pl
import copy


# TODO split in transformer and Seq2SeqModel? an whole left and whole right side for n:m setting.
class Transformer(pl.LightningModule):
    '''A class implementing a transformer used for machine translation.'''
    def __init__(self,
                 src_tokenizer,
                 tgt_tokenizer,
                 src_vocab_size=None,
                 tgt_vocab_size=None,
                 learning_rate=1e-4,
                 weight_decay=0,
                 beta_1=0.9,
                 beta_2=0.98,
                 enable_scheduling=False,
                 warm_up_steps=4000,
                 num_layers=6,
                 d_model=512,
                 dropout=0.1,
                 num_heads=8,
                 d_ff=2048,
                 max_len=128,
                 label_smoothing=0.0,
                 track_bleu=True,
                 track_ter=False,
                 track_tp=False,
                 track_chrf=False,
                 ):
        super().__init__()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.enable_scheduling = enable_scheduling
        self.warm_up_steps = warm_up_steps
        self.d_model = d_model
        
        d_k = d_model // num_heads
        self.max_len = max_len
        self.label_smoothing = label_smoothing

        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.src_vocab_size = src_vocab_size if src_vocab_size is not None else self.src_tokenizer.vocab_size()
        self.tgt_vocab_size = tgt_vocab_size if tgt_vocab_size is not None else self.tgt_tokenizer.vocab_size()

        self.src_embedding = nn.Embedding(self.src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(self.tgt_vocab_size, d_model)
        self.positional_encoder = PositionalEncoder(d_model, max_len)
        self.encoder = Encoder(num_layers, d_model, dropout, num_heads, d_k, d_ff)
        self.decoder = Decoder(num_layers, d_model, dropout, num_heads, d_k, d_ff)
        self.output_linear = nn.Linear(d_model, self.tgt_vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

        # Metrics.
        self.tracked_metrics = ['loss']

        self.track_bleu = track_bleu
        if self.track_bleu:
            self.tracked_metrics.append('bleu')
            self.bleu_metric = torchmetrics.SacreBLEUScore(tokenize='char')

        self.track_ter = track_ter
        if self.track_ter:
            self.tracked_metrics.append('ter')
            self.ter_metric = torchmetrics.TranslationEditRate()

        self.track_tp = track_tp
        if self.track_tp:
            self.tracked_metrics.append('tp')
            self.tp_metric = torch.exp

        self.track_chrf = track_chrf
        if self.track_chrf:
            self.tracked_metrics.append('chrf')
            self.chrf_metric = torchmetrics.CHRFScore()

        self.init_params()

    def forward(self, src_input, tgt_input, e_mask=None, d_mask=None):
        src_input = self.src_embedding(src_input) # (B, L) => (B, L, d_model)
        tgt_input = self.tgt_embedding(tgt_input) # (B, L) => (B, L, d_model)
        src_input = self.positional_encoder(src_input) # (B, L, d_model) => (B, L, d_model)
        tgt_input = self.positional_encoder(tgt_input) # (B, L, d_model) => (B, L, d_model)

        e_output = self.encoder(src_input, e_mask) # (B, L, d_model)
        d_output = self.decoder(tgt_input, e_output, e_mask, d_mask) # (B, L, d_model)

        output = self.softmax(self.output_linear(d_output)) # (B, L, d_model) => # (B, L, tgt_vocab_size)
        return output
    
    # Train.
    def training_step(self, batch, batch_idx):
        logits, tgt_out = self._shared_forward_step(batch, batch_idx)
        loss, metrics = self._shared_eval_step(logits, tgt_out, batch_idx, 'train')
        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.98),
            eps=1e-9,
            weight_decay=self.weight_decay
        )

        if self.enable_scheduling:
            scheduler = {
                'scheduler': WarumUpInverseSquareRootScheduler(optimizer, self.d_model, self.warm_up_steps),
                'interval': 'step',
                'frequency': 1
            }
            return [optimizer], [scheduler]
        else:
            return optimizer

    # Validation.
    def validation_step(self, batch, batch_idx):
        logits, tgt_out = self._shared_forward_step(batch, batch_idx)
        loss, metrics = self._shared_eval_step(logits, tgt_out, batch_idx, 'val')
        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True)
        return metrics

    # Test.
    def test_step(self, batch, batch_idx):
        logits, tgt_out = self._shared_forward_step(batch, batch_idx)
        loss, metrics = self._shared_eval_step(logits, tgt_out, batch_idx, 'test')
        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True)
        return metrics

    # Shared.
    def _shared_forward_step(self, batch, batch_idx):
        src_input, tgt_input, tgt_output = batch
        e_mask, d_mask = self.make_mask(src_input, tgt_input)
        logits = self(src_input, tgt_input, e_mask, d_mask) # (B, L, vocab_size)
        return logits, tgt_output

    def _shared_eval_step(self, logits, tgt_out, batch_idx, context):
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, self.tgt_vocab_size),
            tgt_out.reshape(-1),
            label_smoothing=self.label_smoothing,
            ignore_index=pad_id)
        metrics = {f'{context}_loss': loss.item()}

        if self.track_bleu or self.track_ter or self.track_chrf:
            predictions = self.tgt_tokenizer.Decode(torch.max(logits, dim=2).indices.tolist())
            references = self.tgt_tokenizer.Decode(tgt_out.tolist())
            references = [[r] for r in references]

        if self.track_bleu:
            bleu = self.bleu_metric(predictions, references).item()
            metrics[f'{context}_bleu'] = bleu

        if self.track_ter:
            ter = self.ter_metric(predictions, references).item()
            metrics[f'{context}_ter'] = ter

        if self.track_tp:
            tp = self.tp_metric(loss).item()
            metrics[f'{context}_tp'] = tp
          
        if self.track_chrf:
            chrf = self.chrf_metric(predictions, references).item()
            metrics[f'{context}_chrf'] = chrf

        return loss, metrics

    # Util.
    def encode(self, src, e_mask): # For inference.
        src = self.src_embedding(src.to(device))
        src = self.positional_encoder(src)
        e_output = self.encoder(src, e_mask.to(device)) # (1, L, d_model)
        
        return e_output

    def decode(self, tgt_input, e_output, e_mask, d_mask): # For inference.
        tgt_embedded = self.tgt_embedding(tgt_input.to(device))
        tgt_positional_encoded = self.positional_encoder(tgt_embedded)
        decoder_output = self.decoder(
            tgt_positional_encoded,
            e_output.to(device),
            e_mask.to(device),
            d_mask.to(device)
        ) # (1, L, d_model)

        output = self.softmax(
            self.output_linear(decoder_output)
        ) # (1, L, tgt_vocab_size)

        return output

    def init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def set_dropout(self, dropout, skip_encoder=False, skip_decoder=False):
        if not skip_encoder:
            for encoder_layer in self.encoder.layers:
                encoder_layer.multihead_attention.dropout.p = dropout
                encoder_layer.dropout_1.p = dropout
                encoder_layer.feed_forward.dropout.p = dropout
                encoder_layer.dropout_2.p = dropout

        if not skip_decoder:
            for decoder_layer in self.decoder.layers:
                decoder_layer.masked_multihead_attention.dropout.p = dropout
                decoder_layer.dropout_1.p = dropout
                decoder_layer.multihead_attention.dropout.p = dropout
                decoder_layer.dropout_2.p = dropout
                decoder_layer.feed_forward.dropout.p = dropout
                decoder_layer.dropout_3.p = dropout
    
    def receive_encoder(self, encoder_model):
        '''Sets the encoder of this model to the encoder of the given model.'''
        self.src_embedding = encoder_model.src_embedding
        self.encoder = encoder_model.encoder

    def receive_decoder(self, decoder_model):
        '''Sets the decoder of this model to the decoder of the given model.'''
        self.tgt_embedding = decoder_model.tgt_embedding
        self.decoder = decoder_model.decoder
        self.output_linear = decoder_model.output_linear

    def freeze_encoder(self):
        for param in self.src_embedding.parameters():
            param.requires_grad = False
        for param in self.encoder.parameters():
            param.requires_grad = False

    def freeze_decoder(self):
        for param in self.tgt_embedding.parameters():
            param.requires_grad = False
        for param in self.decoder.parameters():
            param.requires_grad = False
        for param in self.output_linear.parameters():
            param.requires_grad = False

    def make_mask(self, src_input, tgt_input):
        e_mask = (src_input != pad_id).unsqueeze(1)  # (B, 1, L)
        d_mask = (tgt_input != pad_id).unsqueeze(1)  # (B, 1, L)

        nopeak_mask = torch.ones([1, self.max_len, self.max_len], dtype=torch.bool)  # (1, L, L)
        nopeak_mask = torch.tril(nopeak_mask).to(device)  # (1, L, L) to triangular shape
        d_mask = d_mask & nopeak_mask  # (B, L, L) padding false

        return e_mask, d_mask

    def translate(self, src_text, method='greedy', **kwargs): # TODO: if this is made to work with batch sizes greater than 1, some score calculations must be changed!
        self.eval()

        tokenized = self.src_tokenizer.EncodeAsIds(src_text)
        src = torch.LongTensor(pad_or_truncate(tokenized, self.max_len)).unsqueeze(0)
        e_mask = (src != pad_id).unsqueeze(1).to(device) # (1, 1, L)
        e_output = self.encode(src, e_mask) # (1, L, d_model)

        if method == 'greedy':
            translation = self.top_k_top_p_sampling(e_output, e_mask, top_k=1)
        elif method == 'beam':
            translation = self.beam_search(e_output, e_mask,
                kwargs.get('beam_size', 8))
        elif method == 'sampling':
            translation = self.top_k_top_p_sampling(e_output, e_mask,
                kwargs.get('top_k', 0),
                kwargs.get('top_p', 1.0),
                kwargs.get('temperature', 1.0),
            )
        else:
            raise ValueError(f'Unsupported method "{method}."')

        return translation

    def beam_search(self, e_output, e_mask, beam_size):  # TODO beam search should share code with the other inference methods.
        cur_queue = PriorityQueue()
        for k in range(beam_size):
            cur_queue.put(BeamNode(sos_id, -0.0, [sos_id]))
        
        finished_count = 0
        
        for pos in range(self.max_len):
            new_queue = PriorityQueue()
            for k in range(beam_size):
                node = cur_queue.get()
                if node.is_finished:
                    new_queue.put(node)
                else:
                    tgt_input = torch.LongTensor(node.decoded + [pad_id] * (self.max_len - len(node.decoded))).to(device) # (L)
                    d_mask = (tgt_input.unsqueeze(0) != pad_id).unsqueeze(1).to(device) # (1, 1, L)
                    nopeak_mask = torch.ones([1, self.max_len, self.max_len], dtype=torch.bool).to(device)
                    nopeak_mask = torch.tril(nopeak_mask) # (1, L, L) to triangular shape
                    d_mask = d_mask & nopeak_mask # (1, L, L) padding false
                    
                    output = self.decode(
                        tgt_input.unsqueeze(0),
                        e_output,
                        e_mask,
                        d_mask
                    ) # (1, L, tgt_vocab_size)
                    output = torch.topk(output[0][pos], dim=-1, k=beam_size)
                    last_word_ids = output.indices.tolist() # (k)
                    last_word_prob = output.values.tolist() # (k)
                    
                    for i, idx in enumerate(last_word_ids):
                        new_node = BeamNode(idx, -(-node.prob + last_word_prob[i]), node.decoded + [idx])
                        if idx == eos_id:
                            new_node.prob = new_node.prob / float(len(new_node.decoded))
                            new_node.is_finished = True
                            finished_count += 1
                        new_queue.put(new_node)
            
            cur_queue = copy.deepcopy(new_queue)
            
            if finished_count == beam_size:
                break
        
        decoded_output = cur_queue.get().decoded
        
        if decoded_output[-1] == eos_id:
            decoded_output = decoded_output[1:-1]
        else:
            decoded_output = decoded_output[1:]
            
        return self.tgt_tokenizer.decode_ids(decoded_output)

    def top_k_top_p_sampling(self, e_output, e_mask, top_k=0, top_p=1.0, temperature=1.0):
        last_words = torch.LongTensor([pad_id] * self.max_len).to(device) # (L)
        last_words[0] = sos_id # (L)
        cur_len = 1

        for i in range(self.max_len):
            d_mask = (last_words.unsqueeze(0) != pad_id).unsqueeze(1).to(device) # (1, 1, L)
            nopeak_mask = torch.ones([1, self.max_len, self.max_len], dtype=torch.bool).to(device)  # (1, L, L)
            nopeak_mask = torch.tril(nopeak_mask)  # (1, L, L) to triangular shape
            d_mask = d_mask & nopeak_mask  # (1, L, L) padding false

            output = self.decode(
                last_words.unsqueeze(0),
                e_output,
                e_mask,
                d_mask
            ) # (1, L, tgt_vocab_size)

            logits = output[0][i]
            logits = logits / temperature
            filtered_logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
            probabilities = F.softmax(filtered_logits, dim=-1)
            last_word_id = torch.multinomial(probabilities, 1)

            if i < self.max_len-1:
                last_words[i+1] = last_word_id
                cur_len += 1
            
            if last_word_id == eos_id:
                break

        if last_words[-1].item() == pad_id:
            decoded_output = last_words[1:cur_len].tolist()
        else:
            decoded_output = last_words[1:].tolist()
        
        return self.tgt_tokenizer.decode_ids(decoded_output)
    
    # Save and Load.
    def save(self, path):
        torch.save(self.state_dict(), path)
    
    def load(self, path):
        self.load_state_dict(torch.load(path))


class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, dropout, num_heads, d_k, d_ff):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, dropout, num_heads, d_k, d_ff) for i in range(num_layers)])
        self.layer_norm = LayerNormalization(d_model)

    def forward(self, x, e_mask):
        for layer in self.layers:
            x = layer(x, e_mask)
        return self.layer_norm(x)


class Decoder(nn.Module):
    def __init__(self, num_layers, d_model, dropout, num_heads, d_k, d_ff):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model, dropout, num_heads, d_k, d_ff) for i in range(num_layers)])
        self.layer_norm = LayerNormalization(d_model)

    def forward(self, x, e_output, e_mask, d_mask):
        for layer in self.layers:
            x = layer(x, e_output, e_mask, d_mask)
        return self.layer_norm(x)


class BeamNode():
    def __init__(self, cur_idx, prob, decoded):
        self.cur_idx = cur_idx
        self.prob = prob
        self.decoded = decoded
        self.is_finished = False
        
    def __gt__(self, other):
        return self.prob > other.prob
    
    def __ge__(self, other):
        return self.prob >= other.prob
    
    def __lt__(self, other):
        return self.prob < other.prob
    
    def __le__(self, other):
        return self.prob <= other.prob
    
    def __eq__(self, other):
        return self.prob == other.prob
    
    def __ne__(self, other):
        return self.prob != other.prob
    
    def print_spec(self):
        print(f"ID: {self} || cur_idx: {self.cur_idx} || prob: {self.prob} || decoded: {self.decoded}")
    

class PriorityQueue():
    def __init__(self):
        self.queue = []
        
    def put(self, obj):
        heapq.heappush(self.queue, (obj.prob, obj))
        
    def get(self):
        return heapq.heappop(self.queue)[1]
    
    def qsize(self):
        return len(self.queue)
    
    def print_scores(self):
        scores = [t[0] for t in self.queue]
        print(scores)
        
    def print_objs(self):
        objs = [t[1] for t in self.queue]
        print(objs)


def top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float('inf')):
    '''Applies top-K and top-p (nucleus) filtering to the logits.'''

    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear.
    top_k = min(top_k, logits.size(-1))  # Safety check.
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k.
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold.
        sorted_indices_to_remove = cumulative_probs > top_p

        # Shift the indices to the right to keep also the first token above the threshold.
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


def cascaded_inference(batch,
                       src_tokenizer, tgt_tokenizer,
                       src_pvt_model, pvt_tgt_model,
                       score_metric,
                       method = 'greedy',
                       **kwargs):
    '''Performs cascaded inferend with two models.'''

    src_input, tgt_input, tgt_output = batch

    # Convert preprocessed input back to text.
    src_text = src_tokenizer.Decode(src_input.tolist())[0]
    label_text = tgt_tokenizer.Decode(tgt_input.tolist())[0]
    
    # Pass through src-pvt model.
    pvt_text = src_pvt_model.translate(src_text, method=method, kwargs=kwargs)
    
    # Pass through pvt-tgt model.
    tgt_text = pvt_tgt_model.translate(pvt_text, method=method, kwargs=kwargs)
    
    # Calculate metrics.
    score = score_metric.compute(predictions=[tgt_text], references=[[label_text]])['score']

    return score, src_text, pvt_text, tgt_text, label_text

def load_model_from_path(dir, src_tokenizer=None, tgt_tokenizer=None):
    args = load_dict(os.path.join(dir, 'args.json'))
    model = Transformer(
        src_tokenizer,
        tgt_tokenizer,
        src_vocab_size=args.src_vocab_size,
        tgt_vocab_size=args.tgt_vocab_size,
        num_layers=args.num_layers,
        d_model=args.d_model,
        dropout=args.dropout,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        max_len=args.max_len,
    )
    return model
