## for VOC dataset
python train_val.py with dataset=VOC n_shots=1 label_sets=0 batch_size=12 n_iters=50
python train_val.py with dataset=VOC n_shots=1 label_sets=1 batch_size=12 n_iters=50
python train_val.py with dataset=VOC n_shots=1 label_sets=2 batch_size=12 n_iters=50
python train_val.py with dataset=VOC n_shots=1 label_sets=3 batch_size=12 n_iters=30
python train_val.py with dataset=VOC n_shots=5 label_sets=0 batch_size=8 n_iters=30
python train_val.py with dataset=VOC n_shots=5 label_sets=1 batch_size=8 n_iters=30
python train_val.py with dataset=VOC n_shots=5 label_sets=2 batch_size=12 n_iters=30
python train_val.py with dataset=VOC n_shots=5 label_sets=3 batch_size=12 n_iters=30

## for COCO dataset
python train_val.py with dataset=COCO n_shots=1 label_sets=0 batch_size=16 n_iters=50
python train_val.py with dataset=COCO n_shots=1 label_sets=1 batch_size=16 n_iters=50
python train_val.py with dataset=COCO n_shots=1 label_sets=2 batch_size=16 n_iters=50
python train_val.py with dataset=COCO n_shots=1 label_sets=3 batch_size=16 n_iters=50
python train_val.py with dataset=COCO n_shots=5 label_sets=0 batch_size=24 n_iters=30
python train_val.py with dataset=COCO n_shots=5 label_sets=1 batch_size=24 n_iters=30
python train_val.py with dataset=COCO n_shots=5 label_sets=2 batch_size=24 n_iters=30
python train_val.py with dataset=COCO n_shots=5 label_sets=3 batch_size=24 n_iters=30

## for FSS dataset
python train_val.py with dataset=FSS n_shots=1 batch_size=24
python test.py with dataset=FSS mode=test load_snapshot='./runs/API_FSS_align_encoder_sets_0_1way_1shot_train/1/snapshots/best_val.pth'
python train_val.py with dataset=FSS n_shots=5 batch_size=24
python test.py with dataset=FSS mode=test load_snapshot='./runs/API_FSS_align_encoder_sets_0_1way_5shot_train/1/snapshots/best_val.pth'

## for LIDC dataset
python train_val.py with dataset=LIDC n_shots=1 batch_size=16
python test.py with dataset=LIDC mode=test load_snapshot='./runs/API_LIDC_align_encoder_sets_0_1way_1shot_train/1/snapshots/best_val.pth'
