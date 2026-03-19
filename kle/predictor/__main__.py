import sys
import os

# Allow running as both:
#   python -m kle.predictor   (from project root)
#   python -m predictor       (from kle/ directory)
if __package__ in (None, ""):
    # Invoked directly: python predictor/__main__.py  or  python -m predictor from kle/
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from kle.predictor.cli import main
else:
    from .cli import main

main()
