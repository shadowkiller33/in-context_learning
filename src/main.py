import argparse
from ast import Return
from src.utils.eval_utils import sensitivity_compute,cross_entropy
from src.generator import Generator
from src.scorer import Scorer
from src.data_helper import DataHelper
from src.utils.io_utils import print_results, load_pickle, save_pickle
from copy import deepcopy
import logging
import numpy as np
import torch
from scipy.special import entr
from scipy.special import softmax
import random
from torch import nn
import torch.nn.functional as F
import sys

logger = logging.getLogger(__name__)


def main(args):
    default_params = {
        "conditioned_on_correct_classes": True,
        "subsample_test_set": args["subsample_test_set"],
        "api_num_log_prob": args["api_num_log_prob"],
        "approx": args["approx"],
        "bs": args["bs"],
        "mode": args["mode"],
        "data_dir": args["data_dir"],
        "perturbed_num": args["perturbed_num"],
    }
    model = args["models"]
    verbose = args["verbose"]
    # list of all experiment parameters to run
    all_params = []
    for dataset in args["datasets"]:
        for num_shots in args["all_shots"]:
            data_helper = DataHelper(args["data_dir"], dataset)
            for prompt_id, prompt in enumerate(data_helper.get_prompts(dataset)):
                for seed in args["seeds"]:
                    p = deepcopy(default_params)
                    p["prompt_id"] = prompt_id
                    p["prompt"] = prompt
                    p["seed"] = seed
                    p["dataset"] = dataset
                    p["num_shots"] = num_shots
                    all_params.append(p)

    filename = (
        f"{dataset}_{model}_{num_shots}shot_{repr(args['subsample_test_set'])}.pkl"
    )
    if args["use_saved_results"]:
        load_results(args["output_dir"], filename)
    else:
        save_results(all_params, model, args["output_dir"], filename, verbose=verbose)


def load_results(path, filename):
    # load saved results from model
    result_table = load_pickle(path, filename)
    print_results(result_table)




def update_result_dict(table, prompt_id, seed, prompt, entry_name, result):
    if seed not in table:
        prompt_info = {"id": prompt_id, "prompt": prompt, entry_name: result}
        table[seed] = {prompt_id: prompt_info}
    else:
        if prompt_id not in table[seed]:
            table[seed][prompt_id] = {
                "id": prompt_id,
                "prompt": prompt,
                entry_name: result,
            }
        else:
            table[seed][prompt_id][entry_name] = result


def save_results(params_list, model_name, path, filename, verbose=False):
    result_table = {}  # keep all sens, flatness, mi results
    generator = Generator(model_name)
    print(f"Loading model {model_name}")

    for params in params_list:

        if verbose:
            print(
                f"Evaluate on promt id: {params['prompt_id']}, seed: {params['seed']}"
            )
        data_helper = DataHelper(params["data_dir"], params["dataset"])
        scorer = Scorer(params["mode"], params["bs"], generator.get_tokenizer())

        # the current prompt we evaluate metrics on
        prompt_id, prompt = params["prompt_id"], params["prompt"]
        seed = params["seed"]
        ss = ''
        (
            train_sentences,
            train_labels,
            test_sentences,
            test_labels,
        ) = data_helper.get_in_context_prompt(params, ss, seed, verbose=verbose)
        original_raw_resp_test = generator.get_model_response(
            params,  train_sentences=[], train_labels=[], test_sentences=test_sentences
        )
        original_all_label_probs = generator.get_label_probs(
            params, original_raw_resp_test, train_sentences, train_labels, test_sentences
        )
        # content_free_inputs = ["N/A", "", "[MASK]"]
        # p_cf = generator.get_p_content_free(
        #     params,
        #     train_sentences,
        #     train_labels,
        #     content_free_inputs=content_free_inputs,
        # )
        original_acc = scorer.eval_accuracy(
            original_all_label_probs, test_labels  # , mode="diagonal_W", p_cf=p_cf
        )


        # append demos for predictions
        (
            train_sentences,
            train_labels,
            test_sentences,
            test_labels,
        ) = data_helper.get_in_context_prompt(params, prompt, seed, verbose=verbose)
        raw_resp_test = generator.get_model_response(
            params, train_sentences, train_labels, test_sentences
        )
        all_label_probs1 = generator.get_label_probs(
            params, raw_resp_test, train_sentences, train_labels, test_sentences
        )
        xx = all_label_probs1 - original_all_label_probs
        #content_free_inputs = ["N/A", "", "[MASK]"]
        # p_cf = generator.get_p_content_free(
        #     params,
        #     train_sentences,
        #     train_labels,
        #     content_free_inputs=content_free_inputs,
        # )
        acc_1 = scorer.eval_accuracy(
            all_label_probs1, test_labels#, mode="diagonal_W", p_cf=p_cf
        )
        print(f"acc {acc_1}")
        print(f"label probs {all_label_probs1}")



        preds = []
        accs = []
        #kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)
        for i in range(50):
            print(f"iteration count {i}")
            generator = Generator(model_name)
            A = generator.get_model_train_response(
                params, i, 5e-5, train_sentences, train_labels, test_sentences
            )
            all_label_probs = generator.get_label_probs(
                params, A, train_sentences, train_labels, test_sentences
            )
            # p_cf = generator.get_p_content_free(
            #     params,
            #     train_sentences,
            #     train_labels,
            #     content_free_inputs=content_free_inputs,
            # )
            acc_calibrated = scorer.eval_accuracy(all_label_probs, test_labels)
            # acc_calibrated = scorer.eval_accuracy(
            #     all_label_probs, test_labels, mode="diagonal_W", p_cf=p_cf
            # )
            preds.append(all_label_probs - original_all_label_probs)
            accs.append(acc_calibrated)

        cosine = []
        from numpy import dot
        from numpy.linalg import norm
        for x in preds:
            mean = np.mean(x,axis=0)
            mean2 = np.mean(xx,axis=0)
            cos_sim = dot(mean, mean2) / (norm(mean) * norm(mean2))
            cosine.append(cos_sim)
        print(f"logits similarity between two in-context learning and GD(language modeling loss): {cosine}")
        print(f"acc of in-context learning {acc_calibrated}")
        print(f"acc of GD on language modeling loss: {all_label_probs1}")
        x = 1












if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # required arguments
    parser.add_argument(
        "--models",
        dest="models",
        action="store",
        required=True,
        help="name of model(s), e.g., GPT2-XL",
    )
    parser.add_argument(
        "--datasets",
        dest="datasets",
        action="store",
        required=True,
        help="name of dataset(s), e.g., agnews",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save results",
    )
    parser.add_argument(
        "--num_seeds",
        dest="num_seeds",
        action="store",
        required=True,
        help="num seeds for the training set",
        type=int,
    )
    parser.add_argument(
        "--perturbed_num",
        dest="perturbed_num",
        action="store",
        required=True,
        help="num of samples for the perturbed set",
        type=int,
    )
    parser.add_argument(
        "--all_shots",
        dest="all_shots",
        action="store",
        required=True,
        help="num training examples to use",
    )
    parser.add_argument(
        "--mode",
        dest="mode",
        action="store",
        required=True,
        default="max",
        help="the way to calculate flatness",
    )
    # other arguments
    parser.add_argument(
        "--subsample_test_set",
        dest="subsample_test_set",
        action="store",
        required=False,
        type=int,
        default=None,
        help="size of test set to use to speed up eval. None means using all test set",
    )
    parser.add_argument(
        "--api_num_log_prob",
        dest="api_num_log_prob",
        action="store",
        required=False,
        type=int,
        default=100,
        help="number of top tokens to ask for when querying the model. Capped at 100 for OpenAI GPT-3 API",
    )
    parser.add_argument(
        "--bs",
        dest="bs",
        action="store",
        required=False,
        type=int,
        default=None,
        help="batch size for model queries. For OpenAI API, capped at 20. For local running, set this to max out your GPU memory.",
    )
    # flags
    parser.add_argument(
        "--use_saved_results",
        dest="use_saved_results",
        action="store_const",
        const=True,
        default=False,
        help="whether to load the results from pickle files and not run the model",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
    )
    parser.add_argument(
        "--approx",
        dest="approx",
        action="store_const",
        const=True,
        default=False,
        help="whether to set token prob to zero if not in top 100",
    )
    parser.add_argument("--data-dir", required=True, type=str)

    args = parser.parse_args()
    args = vars(args)

    # simple processing
    def convert_to_list(items, is_int=False):
        if is_int:
            return [int(s.strip()) for s in items.split(",")]
        else:
            return [s.strip() for s in items.split(",")]

    args["models"] = convert_to_list(args["models"])[0]
    args["datasets"] = convert_to_list(args["datasets"])
    args["all_shots"] = convert_to_list(args["all_shots"], is_int=True)
    seeds = []
    while len(set(seeds)) < int(args["num_seeds"]):
        seeds.append(random.randint(1, 100))
    args["seeds"] = seeds
    main(args)
