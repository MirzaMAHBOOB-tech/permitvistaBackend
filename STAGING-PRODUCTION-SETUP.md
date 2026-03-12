# PermitVista Environment Setup

## Staging vs Production

Use separate services and separate secrets so testing never touches live customer billing.

| Area | Staging | Production |
|---|---|---|
| Purpose | Dev testing, QA, internal use | Real customers, real payments |
| Frontend URL | `permitvistafrontend.onrender.com` | `permitvista.com` |
| Backend URL | `permitvistabackend.onrender.com` | `api.permitvista.com` |
| Stripe keys | `sk_test_...` | `sk_live_...` |
| Emails | Optional/sandbox | Required, real delivery |

## Render Service Layout

Create four Render services from the same codebase:

1. `permitvista-frontend-staging` (default `onrender.com` domain)
2. `permitvista-frontend-prod` (`permitvista.com`)
3. `permitvista-backend-staging` (default `onrender.com` domain)
4. `permitvista-backend-prod` (`api.permitvista.com`)

## Backend Environment Variables

### Staging backend (`permitvista-backend-staging`)

- `ENVIRONMENT=staging`
- `SHOVELS_API_KEY=<staging or shared key>`
- `GOOGLE_MAPS_API_KEY=<maps key>`
- `STRIPE_SECRET_KEY=sk_test_...`
- `STRIPE_PUBLISHABLE_KEY=pk_test_...`
- `STRIPE_WEBHOOK_SECRET=whsec_...`
- `SENDGRID_API_KEY=` (optional)
- `SENDGRID_FROM_EMAIL=noreply@permitvista.com`
- `FRONTEND_URL=https://permitvistafrontend.onrender.com`
- `BACKEND_URL=https://permitvistabackend.onrender.com`

### Production backend (`permitvista-backend-prod`)

- `ENVIRONMENT=production`
- `SHOVELS_API_KEY=<production key>`
- `GOOGLE_MAPS_API_KEY=<maps key>`
- `STRIPE_SECRET_KEY=sk_live_...`
- `STRIPE_PUBLISHABLE_KEY=pk_live_...`
- `STRIPE_WEBHOOK_SECRET=whsec_...`
- `SENDGRID_API_KEY=SG...` (required)
- `SENDGRID_FROM_EMAIL=noreply@permitvista.com`
- `FRONTEND_URL=https://permitvista.com`
- `BACKEND_URL=https://api.permitvista.com`

## Stripe Webhooks

Configure webhook endpoint for each backend environment:

- `POST /webhook`

Events to subscribe:

- `checkout.session.completed`
- `customer.subscription.created`
- `customer.subscription.updated`
- `customer.subscription.deleted`

Use separate webhook signing secrets for staging and production.

## Email Delivery Behavior

For one-time permit purchases (`$2.99`):

1. Stripe Checkout collects customer email.
2. Backend verifies paid session.
3. Backend generates PDF.
4. Backend starts instant browser download.
5. Backend sends email with PDF attachment through SendGrid.
6. Backend stores purchase in `dbo.customers`.

## Database Tables Used

Subscription status table:

- `dbo.users`

Purchase/email audit table:

- `dbo.customers` with fields:
  - `email`
  - `address_searched`
  - `amount_paid`
  - `stripe_session_id`
  - `created_at`
