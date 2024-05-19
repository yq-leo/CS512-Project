echo "Running on Douban dataset"

@REM python main.py --dataset=Douban --use_attr --alpha=0.1 --runs=5
@REM python main.py --dataset=Douban --use_attr --use_gcn --alpha=0.1 --num_gcn_layers=1 --runs=5
@REM python main.py --dataset=Douban --use_attr --use_gcn --alpha=0.1 --num_gcn_layers=2 --runs=5
@REM python main.py --dataset=Douban --use_attr --use_gcn --alpha=0.1 --num_gcn_layers=3 --runs=5
@REM
@REM python main.py --dataset=Douban --use_attr --alpha=0.3 --runs=5
@REM python main.py --dataset=Douban --use_attr --use_gcn --alpha=0.3 --num_gcn_layers=1 --runs=5
@REM python main.py --dataset=Douban --use_attr --use_gcn --alpha=0.3 --num_gcn_layers=2 --runs=5
@REM python main.py --dataset=Douban --use_attr --use_gcn --alpha=0.3 --num_gcn_layers=3 --runs=5
@REM
@REM python main.py --dataset=Douban --use_attr --alpha=0.5 --runs=5
@REM python main.py --dataset=Douban --use_attr --use_gcn --alpha=0.5 --num_gcn_layers=1 --runs=5
@REM python main.py --dataset=Douban --use_attr --use_gcn --alpha=0.5 --num_gcn_layers=2 --runs=5
@REM python main.py --dataset=Douban --use_attr --use_gcn --alpha=0.5 --num_gcn_layers=3 --runs=5
@REM
@REM python main.py --dataset=Douban --use_attr --alpha=0.7 --runs=5
@REM python main.py --dataset=Douban --use_attr --use_gcn --alpha=0.7 --num_gcn_layers=1 --runs=5
@REM python main.py --dataset=Douban --use_attr --use_gcn --alpha=0.7 --num_gcn_layers=2 --runs=5
@REM python main.py --dataset=Douban --use_attr --use_gcn --alpha=0.7 --num_gcn_layers=3 --runs=5

python main.py --dataset=Douban --use_attr --alpha=0.9 --runs=5
python main.py --dataset=Douban --use_attr --use_gcn --alpha=0.9 --num_gcn_layers=1 --runs=5
python main.py --dataset=Douban --use_attr --use_gcn --alpha=0.9 --num_gcn_layers=2 --runs=5
python main.py --dataset=Douban --use_attr --use_gcn --alpha=0.9 --num_gcn_layers=3 --runs=5

