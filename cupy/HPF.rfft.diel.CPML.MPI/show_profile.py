import sys
import pstats

profile = sys.argv[1]
#show_p = int(sys.argv[2])
#profile = "image_test_profile"
p = pstats.Stats(profile)
#p = pstats.Stats("imag.stats")
p.sort_stats("cumulative")
p.print_stats(0.01)
