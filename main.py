import torch
from time import time
import utility.parser
from NIE_GCN import NIE_GCN
import utility.batch_test
from utility.data_loader import Data
import os

args = utility.parser.parse_args()

utility.batch_test.set_seed(args.seed)

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
os.environ['TORCH_USE_CUDA_DSA'] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

# 读取 one-hot 向量
one_hot_vectors = torch.load('one_hot_vectors.pt')
print(one_hot_vectors.shape)
dataset = Data(args.data_path + args.dataset)
print("4.Init the Recommendation Model:")
Model = NIE_GCN(args, dataset, device, one_hot_vectors)
  
Model.to(device)
print("\tThe recommendation model constructed.")

print("5.Model training and test process:")

Optim = torch.optim.Adam(Model.parameters(), lr=args.lr)
Loss, recall, ndcg = [], [], []
best_report_recall = 0.
best_report_epoch = 0

for epoch in range(args.epochs):
    start_time = time()
  
    if args.layer_att:
        Model.update_attention_A()
        
    if epoch % args.verbose == 0:
        result = utility.batch_test.Test(dataset, Model, device, eval(args.topK), args.multicore, args.test_batch_size)
        if result['recall'][0] > best_report_recall:
            best_report_epoch = epoch + 1
            best_report_recall = result['recall'][0]

        print("recall:", result['recall'], ",precision:", result['precision'], ',ndcg:', result['ndcg'])
    Model.train()

    sample_data = dataset.sample_data_to_train_all()
    users = torch.Tensor(sample_data[:, 0]).long()
    pos_items = torch.Tensor(sample_data[:, 1]).long()
    neg_items = torch.Tensor(sample_data[:, 2]).long()

    users = users.to(device)
    pos_items = pos_items.to(device)
    neg_items = neg_items.to(device)

    users, pos_items, neg_items = utility.batch_test.shuffle(users, pos_items, neg_items)
    num_batch = len(users) // args.batch_size + 1
    
    average_loss = 0.
    average_reg_loss = 0.
    
    for batch_i, (batch_users, batch_positive, batch_negative) in enumerate(utility.batch_test.mini_batch(users, pos_items, neg_items, batch_size=args.batch_size)):
        base_loss, reg_loss = Model.get_bpr_loss(batch_users, batch_positive, batch_negative)
      
        loss = base_loss + reg_loss
        
        Optim.zero_grad()
        loss.backward()
        Optim.step()
        
        average_loss += base_loss.item()
        average_reg_loss += reg_loss.item()
  
    average_loss = average_loss / num_batch
    average_reg_loss = average_reg_loss / num_batch
        
    end_time = time()
   
    print("\t Epoch: %4d| train time: %.3f | train_loss:%.4f + %.4f" % (epoch + 1, end_time - start_time, average_loss, average_reg_loss))

# 創建儲存模型的資料夾
model_save_dir = 'SavedModels'
os.makedirs(model_save_dir, exist_ok=True)
ratio = args.dataset
model_save_path = os.path.join(model_save_dir, f'model_ratio_{ratio[-3:]}.pt')
print(model_save_path)
torch.save(Model, model_save_path)
print("\tModel training process completed.")
