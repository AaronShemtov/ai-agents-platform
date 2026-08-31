# docs

Operational snapshots and notes. Nothing here is applied by anything — it is
reference material for humans.

## tunnel-cmc-oke-backup.json

The remote configuration of Cloudflare Tunnel `cmc-oke`
(`22705624-4a22-4375-8fee-ed74b9b8a7fb`), captured 2026-08-31, at `version: 3`,
before the agent was ever given write access to it.

Why it exists: `PUT /accounts/{account}/cfd_tunnel/{id}/configurations` replaces the
whole ingress list — Cloudflare offers no partial update and keeps no history. If a
read-modify-write goes wrong, this file is the only way back.

To restore:

    curl -X PUT "https://api.cloudflare.com/client/v4/accounts/$ACC/cfd_tunnel/22705624-4a22-4375-8fee-ed74b9b8a7fb/configurations" \
      -H "Authorization: Bearer $CF" -H "Content-Type: application/json" \
      -d '{"config": <the "config" object from the backup>}'

Backups are gitignored: the repository is public and this is operational state, not
source. Re-capture one before any risky change.

## tunnel-homelab-backup.json

`homelab-tunnel` (`12790540-5fc3-4e53-b08e-ce6a1dbb56a5`) at `version: 7`, captured
2026-08-31. **This is the tunnel that serves the websites** — 1ms.my, cv, infra, pwd
and grafana, all routed to the Envoy Gateway. The account has two other tunnels
(`cmc-oke`, `wikijs-aks`); adding a hostname to the wrong one produces a 404 that
looks like a Kubernetes problem.
