# Web reads the Primary Report in the document panel

The living spec said Web should render an authorized Primary Report as the
main answer. A long report in the chat column would replace the conversation
with a document. Web already opens Sources and Files in one `#panel`.

The thread stays the sanitized terminal answer (optional delivery note). When
a Primary Report exists, the owner clicks an affordance on that turn and the
existing panel shows UTF-8 Markdown through the current sanitizer. The panel
does not auto-open on `done`. Report bytes stay in the Blob store; PriorTurns
do not ingest them. Other Published Artifacts stay list-or-download.

This keeps conversation as conversation and still gives Web a primary reading
surface for the report. PKCE and M8 audit stay out of this slice.
