echo "Running on foursquare-twitter dataset"

# Baseline model (BRIGHT)
python main.py --dataset=foursquare-twitter --epochs=400 --lr=5e-4 --runs=5

# Change dist_type
python main.py --dataset=foursquare-twitter --epochs=400 --lr=5e-4 --dist_type=cosine --runs=5

# Change loss function
python main.py --dataset=foursquare-twitter --epochs=400 --lr=5e-4 --dist_type=cosine --loss=Consistency --runs=5
python main.py --dataset=foursquare-twitter --epochs=400 --lr=5e-4 --dist_type=cosine --loss=RegularizedRanking --runs=5
python main.py --dataset=foursquare-twitter --epochs=400 --lr=5e-4 --dist_type=cosine --loss=WeightedRanking --runs=5
python main.py --dataset=foursquare-twitter --epochs=400 --lr=5e-4 --dist_type=cosine --loss=WeightedRegularizedRanking --runs=5

# Change number of layers
python main.py --dataset=foursquare-twitter --epochs=400 --lr=5e-4 --dist_type=cosine --loss=WeightedRegularizedRanking --num_layers=2 --runs=5
python main.py --dataset=foursquare-twitter --epochs=400 --lr=5e-4 --dist_type=cosine --loss=WeightedRegularizedRanking --num_layers=5 --runs=5
python main.py --dataset=foursquare-twitter --epochs=400 --lr=5e-4 --dist_type=cosine --loss=WeightedRegularizedRanking --num_layers=10 --runs=5

# Change mcf_type
python main.py --dataset=foursquare-twitter --epochs=400 --lr=5e-4 --dist_type=cosine --loss=WeightedRegularizedRanking --mcf_type=mean --runs=5
python main.py --dataset=foursquare-twitter --epochs=400 --lr=5e-4 --dist_type=cosine --loss=WeightedRegularizedRanking --mcf_type=max --runs=5
python main.py --dataset=foursquare-twitter --epochs=400 --lr=5e-4 --dist_type=cosine --loss=WeightedRegularizedRanking --mcf_type=min --runs=5
python main.py --dataset=foursquare-twitter --epochs=400 --lr=5e-4 --dist_type=cosine --loss=WeightedRegularizedRanking --mcf_type=sum --runs=5
python main.py --dataset=foursquare-twitter --epochs=400 --lr=5e-4 --dist_type=cosine --loss=WeightedRegularizedRanking --mcf_type=concat --runs=5



