class args:
    method = "DSF_GPR_R"  # DSF_GPR_R, DSF_GPR_I, DSF_Bern_R, DSF_Bern_I, DSF_Jacobi_R, DSF_Jacobi_I
    dataset = "chameleon"
    sub_dataset = ""
    DATAPATH = "data/"
    hpm_opt_mode = False

    # shared-params
    rnd_seed = None
    cpu = False
    hidden_channels = 64
    nepoch = 1000
    early = 100
    dropout = None
    lr = None
    reg = None
    alpha = None

    # PE-params
    pos_enc_type = None
    pe_hid_dim = 64
    pe_indim = None
    pe_normalize = None
    ort_pe_lambda = None
    PE_dropout = None
    PE_alpha = None
    PE_beta = None

    # DSF-GPR/Bern-params
    dprate = None
    Init = None

    # DSF-jacobi-params
    mysole = None
    lr1 = None
    lr3 = None
    lr4 = None
    wd1 = None
    wd3 = None
    wd4 = None
    a = None
    b = None
    dpb = None
    dpt = None
