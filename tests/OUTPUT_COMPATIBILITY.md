# Separate output-compatibility issue: omitted final occupied FFT shell

The FFT spectrum export finds the last occupied bin and then uses
`self.spectrum[:lastindex]`. Because the stop index is exclusive, this omits
that occupied bin rather than just the empty bins after it.

For the existing reference regression, all five samples account for
`5 * 128 * 128 * 126 = 10,321,920` reciprocal-space points. The exported
193 rows account for 10,321,919: occupied bin 193 contains the remaining point.
The four committed references preserve this output behavior.

This is separate from reference-compatible shell assignment. Changing the
slice to include the last occupied bin would change the output row count and
require a deliberate output-format/reference migration. It is deferred here:
the slice, reference files and exact layout/count checks remain unchanged.
