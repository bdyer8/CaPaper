import cProfile
import pstats

cProfile.run(open('2dVideo.py', 'rb'), 'output_stats')

p = pstats.Stats('output_stats')

p.sort_stats('cumulative').print_stats(10)