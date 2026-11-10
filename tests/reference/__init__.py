"""Reference code for the comparison tests.

``reference.eitprocessing`` turns m3resp EIT data into eitprocessing objects
and back, and runs eitprocessing on them (``EITProcessingAdapter``). The
comparison tests use it to check that m3resp gives the results of
eitprocessing 1.8.7. It needs eitprocessing (the ``reference`` extra) and is
not part of the installed package.
"""
