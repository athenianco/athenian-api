# How to delete the account's data

### Discover the IDs

- Athenian account ID `athenian_id`
- Metadata account IDs `meta_ids`

Ways to find those IDs:

- Search in #updates-installations on Slack
- [In JIRA](https://athenianco.atlassian.net/issues/?jql=project%20%3D%20CS%20AND%20issuetype%20%3D%20Customer)
- In state DB by `athenian_id`:

```
select * from account_github_accounts where account_id = <athenian_id>;
```

- In metadata DB by org name:

```
select id from github.accounts where name = '...';
```

### Clear metadata

For each ID in `meta_ids`.

```
python3 athenian/api/hacks/delete_github_account.py metadata <meta_id> >script.sql

PGPASSWORD=password psql -h 0.0.0.0 -p 5432 --db metadata -U production-cloud-sql <script.sql
```

### Clear precomputed

```
python3 athenian/api/hacks/delete_github_account.py precomputed <athenian_id> >script.sql

PGPASSWORD=password psql -h 0.0.0.0 -p 5432 --db precomputed -U production-cloud-sql <script.sql
```

### Clear state

```
python3 athenian/api/hacks/delete_github_account.py state <athenian_id> >script.sql

PGPASSWORD=password psql -h 0.0.0.0 -p 5432 --db state -U production-cloud-sql <script.sql
```
