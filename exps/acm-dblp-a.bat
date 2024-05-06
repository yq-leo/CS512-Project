@REM echo "Running on acm-dblp-a dataset"
@REM
@REM @REM Baseline model (BRIGHT)
@REM python main.py --epochs=250 --lr=1e-3 --runs=5 --gpu
@REM
@REM @REM Change dist_type
@REM python main.py --epochs=250 --lr=1e-3 --dist_type=cosine --runs=5 --gpu

@REM Change loss function
@REM python main.py --epochs=250 --lr=1e-3 --dist_type=cosine --loss=Consistency --runs=3 --gpu
@REM python main.py --epochs=250 --lr=1e-3 --dist_type=cosine --loss=RegularizedRanking --runs=3
@REM python main.py --epochs=250 --lr=1e-3 --dist_type=cosine --loss=WeightedRanking --runs=3 --gpu
@REM python main.py --epochs=250 --lr=1e-3 --dist_type=cosine --loss=WeightedRegularizedRanking --runs=3
@REM
@REM @REM Change number of layers
@REM python main.py --epochs=250 --lr=1e-3 --dist_type=cosine --loss=WeightedRegularizedRanking --num_layers=2 --runs=3
@REM python main.py --epochs=250 --lr=1e-3 --dist_type=cosine --loss=WeightedRegularizedRanking --num_layers=5 --runs=3
@REM python main.py --epochs=250 --lr=1e-3 --dist_type=cosine --loss=WeightedRegularizedRanking --num_layers=10 --runs=3
@REM
@REM @REM Change mcf_type
@REM python main.py --epochs=250 --lr=1e-3 --dist_type=cosine --loss=WeightedRegularizedRanking --mcf_type=mean --runs=3
@REM python main.py --epochs=250 --lr=1e-3 --dist_type=cosine --loss=WeightedRegularizedRanking --mcf_type=max --runs=3
@REM python main.py --epochs=250 --lr=1e-3 --dist_type=cosine --loss=WeightedRegularizedRanking --mcf_type=min --runs=3
@REM python main.py --epochs=250 --lr=1e-3 --dist_type=cosine --loss=WeightedRegularizedRanking --mcf_type=sum --runs=3
@REM python main.py --epochs=250 --lr=1e-3 --dist_type=cosine --loss=WeightedRegularizedRanking --mcf_type=concat --runs=3

python main.py --epochs=250 --lr=1e-3 --dist_type=cosine --loss=RegularizedRanking --runs=3 --lambda_rank=0.3
python main.py --epochs=250 --lr=1e-3 --dist_type=cosine --loss=WeightedRegularizedRanking --runs=3 --lambda_rank=0.3
python main.py --epochs=250 --lr=1e-3 --dist_type=cosine --loss=RegularizedRanking --runs=3 --lambda_rank=0.1
python main.py --epochs=250 --lr=1e-3 --dist_type=cosine --loss=WeightedRegularizedRanking --runs=3 --lambda_rank=0.1
python main.py --epochs=250 --lr=1e-3 --dist_type=cosine --loss=RegularizedRanking --runs=3 --lambda_rank=0.2
python main.py --epochs=250 --lr=1e-3 --dist_type=cosine --loss=WeightedRegularizedRanking --runs=3 --lambda_rank=0.2
python main.py --epochs=250 --lr=1e-3 --dist_type=cosine --loss=RegularizedRanking --runs=3 --lambda_rank=0.1
python main.py --epochs=250 --lr=1e-3 --dist_type=cosine --loss=WeightedRegularizedRanking --runs=3 --lambda_rank=0.1