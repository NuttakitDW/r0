### POST /v3/place_order   (auth header: RCL_TopLevelCheck)

Mandatory body params

| name      | type   | notes                  |
|-----------|--------|------------------------|
| pair      | string | e.g. `BTC/USD`         |
| side      | string | `BUY` or `SELL`        |
| type      | string | `LIMIT` or `MARKET`    |
| quantity  | string | numeric string         |
| timestamp | string | 13‑digit ms epoch      |

Extra **price** is required when `type = LIMIT`.

Success → `200 OK` + JSON shown below. Any rule violation returns `"Success": false` with `ErrMsg`.
