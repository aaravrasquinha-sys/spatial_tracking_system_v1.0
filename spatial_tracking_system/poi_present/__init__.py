"""
poi_present -- Module 5: presentation layer.

Depends on poi_localization's record types (WorldTrackPhaseA /
WorldTrackPhaseB) and nothing lower. Never imports pyslam. Computes
nothing about where people are; it only formats, transports, and
displays what poi_localization (Module 4) already decided.

    poi_localization (M4) --on_tracks(frame, records)--> poi_present
                                                              |
                                                    adapter.to_wire()
                                                              |
                                                     server (WebSocket)
                                                              |
                                                  dashboard (M5a) / VR (M5b)

See SCHEMA.md at the repo root for the wire contract this package
produces, and docs/schema_examples/ for golden messages the tests and
the JS clients both validate against.
"""

__version__ = "0.1.0"
