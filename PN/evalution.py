import seisbench.data as sbd
import seisbench.generate as sbg
from FunctionModule import *
import numpy as np
from PhaseNet_spm import *
#from PhaseNet_cbam import *
import numpy.random as random
import os

cuda_available = torch.cuda.is_available()
device         = torch.device("cuda" if cuda_available else "cpu")

def seed_torch(seed=3407):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch()
model = PhaseNetSPA()
model = load_model_checkpoint("F:\seisbench-main\PN\PhaseNet_sa6_4_iquique_lr1e2_256_best.pth", model)
if torch.cuda.is_available():
    model.cuda()

data = sbd.WaveformDataset("F:\seisbench-main\seisbench\data\datasets\iquique",sampling_rate=100,
                           component_order="ZNE", dimension_order="NCW", cache="full")

train, dev, test = data.train_dev_test()

phase_dict = {
    "trace_p_arrival_sample": "P",
    "trace_pP_arrival_sample": "P",
    "trace_P_arrival_sample": "P",
    "trace_P1_arrival_sample": "P",
    "trace_Pg_arrival_sample": "P",
    "trace_Pn_arrival_sample": "P",
    "trace_PmP_arrival_sample": "P",
    "trace_pwP_arrival_sample": "P",
    "trace_pwPm_arrival_sample": "P",
    "trace_s_arrival_sample": "S",
    "trace_S_arrival_sample": "S",
    "trace_S1_arrival_sample": "S",
    "trace_Sg_arrival_sample": "S",
    "trace_SmS_arrival_sample": "S",
    "trace_Sn_arrival_sample": "S",
}

# train_generator = sbg.GenericGenerator(train)
# dev_generator = sbg.GenericGenerator(dev)
test_generator = sbg.GenericGenerator(test)

augmentations = [
    sbg.WindowAroundSample(list(phase_dict.keys()), samples_before=3000, windowlen=6000, selection="random", strategy="variable"), # or first
    sbg.RandomWindow(windowlen=3001, strategy="pad"),
    sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type="peak"),
    sbg.ChangeDtype(np.float32),
    sbg.ProbabilisticLabeller(label_columns=phase_dict, sigma=30, dim=0)
]

test_generator.add_augmentations(augmentations)

true_positive = 0
false_positive = 0
true_negative = 0
false_negative = 0
event_pred_count = 0
event_true_count = 0
correct_predictions = 0
threshold = 0.3
P_diff = []

for i in range(len(test_generator)):
    sample = test_generator[i]
    label = sample["y"]

    with torch.no_grad():
        pred = model(torch.tensor(sample["X"]).unsqueeze(0).to(device))
        pred = pred[0].cpu().numpy()

    if np.max(pred[0]) >= threshold:
        event_pred = 1
        P_pred = np.argmax(pred[0])
        event_pred_count += 1
    else:
        event_pred = 0
        P_pred = 0  # 判断事件不存在

    if np.max(label[0]) >= threshold:
        event_true = 1
        P_true = np.argmax(label[0])
        event_true_count += 1
    else:
        event_true = 0
        P_true = 0

    p_diff = (P_true - P_pred) / 100
    P_diff.append(p_diff)

    if event_true == 1 and event_pred == 1:
        true_positive += 1
        correct_predictions += 1
    elif event_true == 0 and event_pred == 1:
        false_positive += 1
    elif event_true == 0 and event_pred == 0:
        true_negative += 1
        correct_predictions += 1
    elif event_true == 1 and event_pred == 0:
        false_negative += 1

precision = true_positive / (true_positive + false_positive)
recall = true_positive / (true_positive + false_negative)
f1_score = 2 * precision * recall / (precision + recall)
accuracy = correct_predictions / len(test_generator)

sample_diffs = np.array(P_diff)
time_diffs = sample_diffs
absolute_error_diffs = np.abs(time_diffs)
mae = np.mean(absolute_error_diffs)
mse = np.mean(time_diffs ** 2)
rmse = np.sqrt(mse)

print("MAE:", mae)
print("RMSE:", rmse)
print("Accuracy:", accuracy)
# print("Precision:", precision)
# print("Recall:", recall)
print("F1-score:", f1_score)
# print("Event prediction count:", event_pred_count)
# print("Event true count:", event_true_count)