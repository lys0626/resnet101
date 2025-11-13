import torch
import os
import torch.utils.data
from torch.cuda.amp import autocast

# --- 从您的项目中导入 ---
from utilities import utils, utils_ddp, metric
from engine import Engine 
from main import parser  # <--- 重用 main.py 的参数解析器
from utilities.utils import get_transform
opj = os.path.join

# (根据需要从您的数据集中导入)
from utilities.coco import COCO2014
from utilities.voc import VOC2007
from utilities.nih import nihchest


def run_evaluation(args):
    """
    运行模型评估的主函数
    """
    
    # 1. 检查是否指定了 checkpoint
    if not args.resume:
        if args.rank == 0:
            print("错误: 必须使用 -r /path/to/best_checkpoint.pt 来指定要测试的模型。")
        return

    # 2. 将 'evaluate' 模式设为 0 (纯评估模式)
    args.evaluate = 0
    
    # 3. 创建 Engine 实例
    #    (这将创建模型并使用 args.resume 加载 checkpoint)
    if args.rank == 0:
        print(f"--- 正在初始化引擎并从 {args.resume} 加载模型 ---")
    
    engine = Engine(args) 
    model = engine.model      # 获取已加载权重并封装 DDP 的模型
    loss_fn = engine.loss_fn  # 获取损失函数

    # 4. --- 手动加载真正的 "测试" 数据集 ---
    if args.rank == 0:
        print(f"--- 正在为 '{args.data_set}' 加载 'test' 模式数据集 ---")
        
    test_transfm = get_transform(args, is_train=False)
    
    data_dict = {'MS-COCO': COCO2014, 'VOC2007': VOC2007, 'NIH-CHEST': nihchest}
    data_dir = args.data_root 
    
    if args.data_set in ('MS-COCO'):
        data_dir = opj(args.data_root, 'COCO2014')
        real_test_set = data_dict[args.data_set](data_dir, phase='val', transform=test_transfm)
    elif args.data_set in ('VOC2007'):
        data_dir = opj(args.data_root, 'VOC2007')
        real_test_set = data_dict[args.data_set](data_dir, phase='test', transform=test_transfm)
    elif args.data_set in ('NIH-CHEST'):
        real_test_set = data_dict[args.data_set](data_dir, mode='test', transform=test_transfm) # <--- 显式使用 'test'
    else:
        raise ValueError(f"数据集 {args.data_set} 未在 test.py 中配置")

    # 5. --- 创建 "测试" Dataloader ---
    if args.distributed:
        test_sampler = torch.utils.data.distributed.DistributedSampler(real_test_set, shuffle=False)
    else:
        test_sampler = None
    
    real_test_loader = torch.utils.data.DataLoader(
        real_test_set, 
        batch_size=args.batch_size_per,
        num_workers=args.num_workers, 
        pin_memory=True, 
        drop_last=False, 
        sampler=test_sampler,
        shuffle=False
    )
    
    if args.rank == 0:
        print(f"成功加载 {len(real_test_set)} 个测试图像。")

    # 6. --- 手动运行评估循环 ---
    model.eval()
    ap_meter = metric.AveragePrecisionMeter()
    loss_meter = metric.AverageMeter('loss_test')

    if args.rank == 0:
        from tqdm import tqdm
        loader_tqdm = tqdm(real_test_loader, desc="Testing")
    else:
        loader_tqdm = real_test_loader

    for i, data in enumerate(loader_tqdm):
        # (来自 engine.on_start_batch)
        inputs = data['image'].to(args.rank)
        targets_gt = data['target']
        file_name = data['name']
        targets = targets_gt.clone().to(args.rank)
        targets[targets == -1] = 0

        # (来自 engine.on_forward)
        with torch.no_grad():
            with autocast(enabled=not args.disable_amp):
                outputs = model(inputs, args={}) 
                loss = loss_fn(outputs, targets)
        
        outputs = outputs[0][:inputs.shape[0]].data if type(outputs) == tuple else outputs[:inputs.shape[0]].data

        # (来自 engine.on_end_batch)
        # c. DDP 收集 (来自 engine.on_end_batch)
        if args.distributed:
            outputs_all = utils_ddp.distributed_concat(outputs.detach(), args.batch_size)
            targets_gt_all = utils_ddp.distributed_concat(targets_gt.detach().to(args.rank), args.batch_size)
            loss_all = utils_ddp.distributed_concat(loss.detach().unsqueeze(0), args.world_size)
        else:
            # 单 GPU 模式: _all 版本就是原始张量
            outputs_all = outputs.detach()
            targets_gt_all = targets_gt.detach() # targets_gt 已经在 CPU 上
            loss_all = loss.detach().cpu().mean()
        loss_meter.add(loss.cpu())
        if args.rank == 0:
            ap_meter.add(outputs_all.detach().cpu(), targets_gt_all.cpu(), file_name)
    
    if args.distributed:
        utils_ddp.barrier() # 等待所有进程完成

    # 7. --- 在主进程中打印最终结果 ---
    if args.rank == 0:
        loss = loss_meter.average()
        (mAP, AP) = ap_meter.mAP()
        OP, OR, OF1, CP, CR, CF1 = ap_meter.overall()
        
        str_precision = f'OF1:{OF1:.2f}, CF1:{CF1:.2f}, mAP:{mAP:.4f}'
        print(f"\n--- [ 最终测试结果 ] ---")
        print(f"Checkpoint: {args.resume}")
        print(f"Loss: {loss:.4f}, {str_precision}")
        
        OP_k, OR_k, OF1_k, CP_k, CR_k, CF1_k = ap_meter.overall_topk(args.acc_top_k)
        str_verbose = f'\n详细指标 (Top-k k={args.acc_top_k}):\n' \
                      f'OP: {OP:.2f}, OR: {OR:.2f}, OF1: {OF1:.2f}, CP: {CP:.2f}, CR: {CR:.2f}, CF1: {CF1:.2f}\n' \
                      f'OP_{args.acc_top_k}: {OP_k:.2f}, OR_{args.acc_top_k}: {OR_k:.2f}, OF1_{args.acc_top_k}: {OF1_k:.2f}\n' \
                      f'CP_{args.acc_top_k}: {CP_k:.2f}, CR_{args.acc_top_k}: {CR_k:.2f}, CF1_{args.acc_top_k}: {CF1_k:.2f}\n'
        print(str_verbose)
        print(f"所有类别的 AP:\n{AP}")

    if args.distributed:
        utils_ddp.cleanup()

if __name__ == "__main__":
    # 1. 解析参数
    args = parser.parse_args()

    # 2. 初始化环境 (自动检测 DDP 或单 GPU)
    args = utils.init(args) 
    
    # 3. 运行评估
    run_evaluation(args)