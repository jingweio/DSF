from parse import args
from main import model_run, keep2decimal
import numpy as np
import optuna

args.method = "DSF_GPR_R"
args.dataset = "chameleon"
args.sub_dataset = ""
args.hpm_opt_mode = True

opt_logfile = open(f"{args.method}_{args.dataset}_{args.sub_dataset}_hpmOpt.txt", "w+")
max_evals = 200
trail_step = 0


def log_trial(str, file, end="\r\n"):
    print(str, end=end)
    print(str, file=file, end=end)
    file.flush()


def get_rnd_seed():
    # return a random seed in 6 numbers
    return np.random.randint(99999, 999999)


def cut(v):
    if int(v) == v:
        return int(v)
    else:
        return float(keep2decimal(v))


def objective(trial):
    global trail_step
    trail_step += 1
    # sampling space
    args.reg = cut(trial.suggest_uniform("reg", 5e-8, 1e-2))
    args.lr = cut(trial.suggest_uniform("lr", 1e-4, 0.1))
    args.dropout = cut(trial.suggest_float("dropout", 0, 0.8, step=0.1))
    args.dprate = cut(trial.suggest_float("dprate", 0, 0.8, step=0.1))
    args.Init = trial.suggest_categorical("Init", ['PPR', 'NPPR', 'Random'])
    args.alpha = trial.suggest_categorical("alpha", [0.1, 0.2, 0.5, 0.9])
    args.pos_enc_type = trial.suggest_categorical("pos_enc_type", ["RW_PE", "LAP_PE"])
    args.pe_indim = cut(trial.suggest_int("pe_indim", 2, 32, step=2))
    args.pe_normalize = trial.suggest_categorical("pe_normalize", [True, False])
    args.PE_dropout = cut(trial.suggest_float("PE_dropout", 0, 0.8, step=0.1))
    args.PE_alpha = cut(trial.suggest_float("PE_alpha", 0, 1, step=0.1))
    args.ort_pe_lambda = cut(trial.suggest_uniform("ort_pe_lambda", 1e-2, 1))
    # run model
    args.rnd_seed = get_rnd_seed()
    val_acc, tst_acc = model_run(args)
    # log result
    hyp_str = f"reg={args.reg}, lr={args.lr}, dropout={args.dropout}, dprate={args.dprate}, Init={args.Init}, alpha={args.alpha}, " \
              f"pos_enc_type={args.pos_enc_type}, pe_indim={args.pe_indim}, pe_normalize={args.pe_normalize}, " \
              f"PE_dropout={args.PE_dropout}, PE_alpha={args.PE_alpha}, ort_pe_lambda={args.ort_pe_lambda}"
    log_trial(f"trail={trail_step}/{max_evals}, seed={args.rnd_seed}, "
              f"val-acc={keep2decimal(val_acc * 100)}%, tst-acc={keep2decimal(tst_acc * 100)}% @ {hyp_str}", file=opt_logfile)
    return val_acc


def main():
    log_trial(f"Tuning Hyper-params of {args.method} on dataset-{args.dataset}_{args.sub_dataset} in {max_evals} evals", file=opt_logfile)
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(), direction="maximize")
    study.optimize(objective, n_trials=max_evals)
    best_trial = study.best_trial
    args.pos_enc_type = best_trial.params["pos_enc_type"]
    args.Init = best_trial.params["Init"]
    args.alpha = best_trial.params["alpha"]
    args.pe_normalize = best_trial.params["pe_normalize"]
    args.reg = cut(best_trial.params["reg"])
    args.lr = cut(best_trial.params["lr"])
    args.dropout = cut(best_trial.params["dropout"])
    args.dprate = cut(best_trial.params["dprate"])
    args.PE_dropout = cut(best_trial.params["PE_dropout"])
    args.PE_alpha = cut(best_trial.params["PE_alpha"])
    args.pe_indim = cut(best_trial.params["pe_indim"])
    args.ort_pe_lambda = cut(best_trial.params["ort_pe_lambda"])
    final_hyp_str = f"reg={args.reg}, lr={args.lr}, dropout={args.dropout}, dprate={args.dprate}, Init={args.Init}, alpha={args.alpha}, " \
                    f"pos_enc_type={args.pos_enc_type}, pe_indim={args.pe_indim}, pe_normalize={args.pe_normalize}, " \
                    f"PE_dropout={args.PE_dropout}, PE_alpha={args.PE_alpha}, ort_pe_lambda={args.ort_pe_lambda}"
    log_str = f"Final-Selection:\r\n{final_hyp_str}\r\n" \
              f"best-val-acc={keep2decimal(best_trial.value * 100)}%"
    log_trial(log_str, file=opt_logfile)
    opt_logfile.close()


if __name__ == '__main__':
    main()
