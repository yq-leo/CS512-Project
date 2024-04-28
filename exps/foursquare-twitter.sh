#echo "Running on foursquare-twitter dataset"
#
## Baseline model (BRIGHT)
#python main.py --dataset=foursquare-twitter --epochs=400 --lr=5e-4
#
## Change dist_type
#python main.py --dataset=foursquare-twitter --epochs=400 --lr=5e-4 --dist_type=cosine
#
## Change loss function
#python main.py --dataset=foursquare-twitter --epochs=400 --lr=5e-4 --dist_type=cosine --loss=Consistency
#python main.py --dataset=foursquare-twitter --epochs=400 --lr=5e-4 --dist_type=cosine --loss=RegularizedRanking
#python main.py --dataset=foursquare-twitter --epochs=400 --lr=5e-4 --dist_type=cosine --loss=WeightedRanking
#python main.py --dataset=foursquare-twitter --epochs=400 --lr=5e-4 --dist_type=cosine --loss=WeightedRegularizedRanking
#
## Change number of layers
#python main.py --dataset=foursquare-twitter --epochs=400 --lr=5e-4 --dist_type=cosine --loss=WeightedRegularizedRanking --num_layers=2
#python main.py --dataset=foursquare-twitter --epochs=400 --lr=5e-4 --dist_type=cosine --loss=WeightedRegularizedRanking --num_layers=5
#python main.py --dataset=foursquare-twitter --epochs=400 --lr=5e-4 --dist_type=cosine --loss=WeightedRegularizedRanking --num_layers=10
#
## Change mcf_type
#python main.py --dataset=foursquare-twitter --epochs=400 --lr=5e-4 --dist_type=cosine --loss=WeightedRegularizedRanking --mcf_type=mean
#python main.py --dataset=foursquare-twitter --epochs=400 --lr=5e-4 --dist_type=cosine --loss=WeightedRegularizedRanking --mcf_type=max
python main.py --dataset=foursquare-twitter --epochs=400 --lr=5e-4 --dist_type=cosine --loss=WeightedRegularizedRanking --mcf_type=min
python main.py --dataset=foursquare-twitter --epochs=400 --lr=5e-4 --dist_type=cosine --loss=WeightedRegularizedRanking --mcf_type=sum
python main.py --dataset=foursquare-twitter --epochs=400 --lr=5e-4 --dist_type=cosine --loss=WeightedRegularizedRanking --mcf_type=concat

