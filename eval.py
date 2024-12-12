import torch
import utility.parser
from NIE_GCN import NIE_GCN
from utility.data_loader import Data
import utility.batch_test
import os


def run_eval():
    # 解析參數
    args = utility.parser.parse_args()
    print(args)

    # 設定隨機種子
    utility.batch_test.set_seed(args.seed)

    # 設定 GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

    # 讀取 one-hot 向量
    one_hot_vectors = torch.load('one_hot_vectors.pt')
    print(one_hot_vectors.shape)

    # 載入資料集
    dataset = Data(args.data_path + args.dataset)

    # 初始化推薦模型
    print("Init the Recommendation Model:")

    # 載入已訓練的模型參數
    model = torch.load(f'SavedModels/model_ratio_1.0.pt', weights_only=False)
    # Model.eval()  # 設置模型為評估模式

    print("The recommendation model loaded and set to evaluation mode.")

    # 進行評估
    print("Model evaluation process:")
    result, recommended_items = utility.batch_test.Test(dataset, model, device, eval(args.topK), args.multicore, args.test_batch_size, return_recommendations=True)

    # 輸出評估結果
    print("Evaluation results:")
    print("Recall:", result['recall'])
    print("Precision:", result['precision'])
    print("NDCG:", result['ndcg'])

    # 根據 dataset 生成輸出檔案名稱
    recommendations_file = f"result/recommended_items_{args.dataset}.txt"
    evaluation_file = f"result/evaluation_results_{args.dataset}.txt"
    print(recommended_items)
    # 將推薦的 items 寫入文件
    with open(recommendations_file, 'w') as f:
        for user, items in recommended_items.items():
            f.write(f"{items}\n")

    # print(f"Recommended items have been written to {recommendations_file}")

    # 將評估結果寫入文件
    # with open(evaluation_file, 'w') as f:
    #     f.write("Evaluation results:\n")
    #     f.write(f"Recall: {result['recall']}\n")
    #     f.write(f"Precision: {result['precision']}\n")
    #     f.write(f"NDCG: {result['ndcg']}\n")
    return recommended_items
# print(f"Evaluation results have been written to {evaluation_file}")
run_eval()