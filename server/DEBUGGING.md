# How to debug a problem with a specific account

### Discover the account's users

```
select * from user_accounts where account_id = <id>;
```

### Proxy the production DB

```
cloud_sql_proxy -instances=<GCP project>:us-east1:<see instance name in Cloud SQL console>=tcp:5432
```

### (optional) Proxy memcached

```
gcloud container clusters get-credentials production-cluster --zone us-east1-c --project <GCP project>
kubectl port-forward service/mcrouter-mcrouter 11211:5000 -n mcrouter
```

### Launch API on the remote DB

```
source .env  # export all required the environment variables
python3 -m athenian.api \
  --metadata-db=postgresql://production-cloud-sql:<password>@0.0.0.0:5432/metadata \
  --state-db=postgresql://production-cloud-sql:<password>@0.0.0.0:5432/state  \
  --precomputed-db=postgresql://production-cloud-sql:<password>@0.0.0.0:5432/precomputed \
  --persistentdata-db=postgresql://production-cloud-sql:<password>@0.0.0.0:5432/persistentdata \
  --port 8081 --force-user='<user ID that you've found earlier>'
```

**This is dangerous. Remember that you are working with the production database, this is a big responsibility Please think twice about the local changes in the code to not screw the DB up.**

### Reproduce the reported case

```
curl http://0.0.0.0:8081/v1/endpoint --data '{...}' | jq
```