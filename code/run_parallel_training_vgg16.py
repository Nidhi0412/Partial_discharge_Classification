"""Compatibility entrypoint; one shared, manifest-driven implementation.

Historical implementation is preserved in audit/before_pipeline_fixes_20261103.
Run this file with --help for the explicit workflow. No automatic data mutation.
"""
from training_pipeline import main


if __name__ == "__main__":
    main(default_model="vgg16")
