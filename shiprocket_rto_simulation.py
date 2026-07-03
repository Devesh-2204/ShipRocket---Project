# -*- coding: utf-8 -*-
"""
=============================================================================
BUYER RELIABILITY INTELLIGENCE FRAMEWORK
Decoding RTO / NDR / Ghost-Customer Behavior — Shiprocket Buyer Journey
=============================================================================
MODULE 1: Customer Profile + Order Event Generator

Logic:
  - 50,000 unique customers
  - Each customer generates 1-8 ORDER events over a year (order-based persona)
  - Persona drives payment mode, RTO risk, ghost behavior, return behavior
  - Risk score is recomputed at Gate 1 (order creation) and Gate 2 (pre-delivery)
  - Output: one row per order, with order-level AND customer-cumulative features

Author: [Your Name]
Project: Product Analytics Internship — Shiprocket
=============================================================================
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
random.seed(42)

# =============================================================================
# SECTION 1: PERSONA DEFINITIONS
# Each persona = a distinct buyer reliability archetype, not a demographic
# segment alone. This is the core design decision: RTO/NDR/Ghost behavior is
# driven by RELIABILITY type, which correlates with but is not identical to
# demographics.
# =============================================================================

PERSONAS = {

    'P1_Prepaid_Loyalist': {
        'weight': 0.18,
        'city_tier_dist': {'metro': 0.55, 'tier2': 0.32, 'tier3': 0.13},
        'payment_pref': {'prepaid': 0.85, 'cod': 0.15},
        'category_dist': {'electronics': 0.30, 'home': 0.25, 'beauty': 0.20,
                           'fashion': 0.15, 'grocery': 0.10},
        'orders_per_year': (2, 8),
        'base_rto_rate': 0.04,          # very low, prepaid + repeat trust
        'base_return_rate': 0.06,
        'ghost_propensity': 0.01,
        'phone_pickup_rate': 0.92,
        'address_quality_mean': 0.90,   # 0-1 score, higher = cleaner address
        'avg_order_value': 1400,
    },

    'P2_COD_Occasional': {
        'weight': 0.20,
        'city_tier_dist': {'metro': 0.40, 'tier2': 0.40, 'tier3': 0.20},
        'payment_pref': {'prepaid': 0.30, 'cod': 0.70},
        'category_dist': {'fashion': 0.35, 'beauty': 0.20, 'home': 0.20,
                           'electronics': 0.15, 'grocery': 0.10},
        'orders_per_year': (1, 4),
        'base_rto_rate': 0.18,
        'base_return_rate': 0.12,
        'ghost_propensity': 0.06,
        'phone_pickup_rate': 0.75,
        'address_quality_mean': 0.72,
        'avg_order_value': 900,
    },

    'P3_Tier3_PriceSensitive': {
        'weight': 0.16,
        'city_tier_dist': {'metro': 0.05, 'tier2': 0.30, 'tier3': 0.65},
        'payment_pref': {'prepaid': 0.15, 'cod': 0.85},
        'category_dist': {'fashion': 0.40, 'home': 0.20, 'beauty': 0.15,
                           'grocery': 0.15, 'electronics': 0.10},
        'orders_per_year': (1, 3),
        'base_rto_rate': 0.30,          # structurally higher — courier + address friction
        'base_return_rate': 0.10,
        'ghost_propensity': 0.10,
        'phone_pickup_rate': 0.60,
        'address_quality_mean': 0.55,   # landmark-based addresses, weaker pin accuracy
        'avg_order_value': 650,
    },

    'P4_Fashion_SerialReturner': {
        'weight': 0.10,
        'city_tier_dist': {'metro': 0.55, 'tier2': 0.30, 'tier3': 0.15},
        'payment_pref': {'prepaid': 0.40, 'cod': 0.60},
        'category_dist': {'fashion': 0.90, 'beauty': 0.10},
        'orders_per_year': (3, 8),
        'base_rto_rate': 0.12,
        'base_return_rate': 0.38,       # the defining trait — accepts delivery, then returns
        'ghost_propensity': 0.03,
        'phone_pickup_rate': 0.85,
        'address_quality_mean': 0.85,
        'avg_order_value': 1100,
        # listing-quality sensitivity: this persona's return rate responds to photo/size-chart presence
        'return_sensitive_to_listing_quality': True,
    },

    'P5_Ghost_Customer': {
        'weight': 0.07,
        'city_tier_dist': {'metro': 0.25, 'tier2': 0.35, 'tier3': 0.40},
        'payment_pref': {'prepaid': 0.05, 'cod': 0.95},   # ghosts overwhelmingly choose COD
        'category_dist': {'fashion': 0.45, 'electronics': 0.25, 'beauty': 0.15,
                           'home': 0.10, 'grocery': 0.05},
        'orders_per_year': (1, 5),
        'base_rto_rate': 0.55,
        'base_return_rate': 0.05,
        'ghost_propensity': 0.65,       # most of their orders show ghost behavior
        'phone_pickup_rate': 0.20,
        'address_quality_mean': 0.65,   # address often fine — it's intent, not accuracy, that's the issue
        'avg_order_value': 850,
    },

    'P6_Festive_BulkBuyer': {
        'weight': 0.12,
        'city_tier_dist': {'metro': 0.45, 'tier2': 0.35, 'tier3': 0.20},
        'payment_pref': {'prepaid': 0.55, 'cod': 0.45},
        'category_dist': {'home': 0.30, 'fashion': 0.25, 'electronics': 0.20,
                           'beauty': 0.15, 'grocery': 0.10},
        'orders_per_year': (2, 6),
        'base_rto_rate': 0.10,
        'base_return_rate': 0.09,
        'ghost_propensity': 0.02,
        'phone_pickup_rate': 0.85,
        'address_quality_mean': 0.82,
        'avg_order_value': 1600,
        'festive_multiplier_on_orders': True,
    },

    'P7_FirstTime_ColdStart': {
        'weight': 0.17,
        'city_tier_dist': {'metro': 0.35, 'tier2': 0.35, 'tier3': 0.30},
        'payment_pref': {'prepaid': 0.20, 'cod': 0.80},   # new customers default to COD (trust gap)
        'category_dist': {'fashion': 0.35, 'beauty': 0.20, 'home': 0.20,
                           'electronics': 0.15, 'grocery': 0.10},
        'orders_per_year': (1, 1),      # by definition, this is their only order this year
        'base_rto_rate': 0.24,          # higher uncertainty, no history to smooth it
        'base_return_rate': 0.11,
        'ghost_propensity': 0.08,
        'phone_pickup_rate': 0.68,
        'address_quality_mean': 0.68,
        'avg_order_value': 800,
    },
}

CATEGORY_RTO_MULTIPLIER = {
    'fashion': 1.35, 'beauty': 1.05, 'home': 0.85,
    'electronics': 0.60, 'grocery': 0.50,
}

FESTIVAL_WINDOWS = [
    {'name': 'Republic Day', 'start': '2024-01-22', 'end': '2024-01-28', 'mult': 1.2},
    {'name': 'Summer Sale', 'start': '2024-05-10', 'end': '2024-06-25', 'mult': 1.3},
    {'name': 'Independence Day', 'start': '2024-08-10', 'end': '2024-08-17', 'mult': 1.2},
    {'name': 'Diwali', 'start': '2024-10-28', 'end': '2024-11-06', 'mult': 1.9},
    {'name': 'Year-End Sale', 'start': '2024-12-18', 'end': '2025-01-02', 'mult': 1.7},
]

def get_season(order_date):
    for w in FESTIVAL_WINDOWS:
        s, e = datetime.strptime(w['start'], '%Y-%m-%d'), datetime.strptime(w['end'], '%Y-%m-%d')
        if s <= order_date <= e:
            return w['name'], w['mult']
    return 'Off-Season', 1.0


# =============================================================================
# SECTION 2: CUSTOMER GENERATOR
# =============================================================================

def generate_customer(customer_id, persona_key, persona):
    city_tier = np.random.choice(list(persona['city_tier_dist'].keys()),
                                  p=list(persona['city_tier_dist'].values()))
    address_quality = float(np.clip(
        np.random.normal(persona['address_quality_mean'], 0.12), 0.05, 1.0
    ))
    return {
        'customer_id': customer_id,
        'persona': persona_key,
        'city_tier': city_tier,
        'address_quality_score': round(address_quality, 2),
        'phone_pickup_rate': persona['phone_pickup_rate'],
    }


# =============================================================================
# SECTION 3: ORDER EVENT GENERATOR
# Generates each order, computes Gate-1 and Gate-2 outcomes sequentially,
# and updates the customer's cumulative history as it goes (so ghost
# detection emerges from behavior over time, not a static label).
# =============================================================================

def generate_orders_for_customer(customer, persona_key, persona):
    orders = []
    n_orders = random.randint(*persona['orders_per_year'])
    order_days = sorted(random.sample(range(1, 360), n_orders))

    # running history — this is what a real Gate-1 model would query
    hist_orders = 0
    hist_rto = 0
    hist_returns = 0
    hist_cod = 0

    for i, day in enumerate(order_days):
        order_date = datetime(2024, 1, 1) + timedelta(days=day)
        season, demand_mult = get_season(order_date)

        category = np.random.choice(list(persona['category_dist'].keys()),
                                     p=list(persona['category_dist'].values()))
        payment_mode = np.random.choice(list(persona['payment_pref'].keys()),
                                        p=list(persona['payment_pref'].values()))

        order_value = int(persona['avg_order_value'] * demand_mult * random.uniform(0.75, 1.4))

        # ---------------- GATE 1: order-creation risk ----------------
        # cold-start fallback: first order for this customer relies more on
        # city-tier/category priors than on (nonexistent) personal history
        if hist_orders == 0:
            personal_rto_signal = persona['base_rto_rate']  # prior only
        else:
            personal_rto_signal = 0.5 * persona['base_rto_rate'] + 0.5 * (hist_rto / hist_orders)

        cat_mult = CATEGORY_RTO_MULTIPLIER.get(category, 1.0)
        cod_mult = 1.8 if payment_mode == 'cod' else 1.0
        gate1_rto_risk = float(np.clip(personal_rto_signal * cat_mult * cod_mult, 0.01, 0.95))

        # ---------------- GATE 2: pre-delivery engagement ----------------
        picked_up_call = random.random() < customer['phone_pickup_rate']
        engaged_with_notification = random.random() < (0.55 if picked_up_call else 0.20)

        # ghost behavior draw — persona-driven, dampened by engagement
        is_ghost_event = (
            random.random() < persona['ghost_propensity'] and not picked_up_call
        )

        # address-driven NDR (independent of ghost behavior)
        address_ndr = random.random() < (1 - customer['address_quality_score']) * 0.25

        ndr_triggered = is_ghost_event or address_ndr or (random.random() < gate1_rto_risk * 0.6)

        if ndr_triggered:
            if is_ghost_event:
                ndr_reason = 'Customer unreachable / refused'
            elif address_ndr:
                ndr_reason = 'Address issue'
            else:
                ndr_reason = 'Delivery attempt failed (other)'

            # reattempt success depends on engagement signal from Gate 2
            reattempt_success_prob = 0.65 if engaged_with_notification else 0.25
            delivered_on_reattempt = random.random() < reattempt_success_prob
            rto_triggered = not delivered_on_reattempt
            delivered = delivered_on_reattempt
        else:
            ndr_reason = None
            rto_triggered = False
            delivered = True

        # ---------------- Post-delivery return (only if delivered) ----------------
        post_delivery_return = False
        return_reason = None
        if delivered:
            return_prob = persona['base_return_rate']
            if persona.get('return_sensitive_to_listing_quality'):
                # simulate listing quality as a per-order random draw (proxy: has real photos)
                has_customer_photos = random.random() < 0.45
                if not has_customer_photos:
                    return_prob *= 1.6
                else:
                    return_prob *= 0.7
            if category == 'fashion':
                return_prob *= 1.3
            post_delivery_return = random.random() < return_prob
            if post_delivery_return:
                return_reason = np.random.choice(
                    ['Size/fit mismatch', 'Quality gap vs listing', 'Changed mind', 'Damaged in transit'],
                    p=[0.40, 0.25, 0.25, 0.10]
                )

        # cost impact
        if rto_triggered:
            non_resalable = category in ('beauty', 'grocery')
            logistics_cost = random.randint(150, 400)
            cogs_writeoff = order_value if non_resalable else 0
            financial_loss = logistics_cost + cogs_writeoff
        else:
            financial_loss = 0

        orders.append({
            'order_id': f"{customer['customer_id']}-{i+1}",
            'customer_id': customer['customer_id'],
            'persona': persona_key,
            'order_num_for_customer': i + 1,
            'order_date': order_date.strftime('%Y-%m-%d'),
            'season': season,
            'city_tier': customer['city_tier'],
            'category': category,
            'payment_mode': payment_mode,
            'order_value_inr': order_value,
            'address_quality_score': customer['address_quality_score'],
            'gate1_rto_risk_score': round(gate1_rto_risk, 3),
            'picked_up_pre_delivery_call': picked_up_call,
            'engaged_with_notification': engaged_with_notification,
            'ndr_triggered': ndr_triggered,
            'ndr_reason': ndr_reason,
            'rto_triggered': rto_triggered,
            'delivered': delivered,
            'post_delivery_return': post_delivery_return,
            'return_reason': return_reason,
            'is_ghost_event': is_ghost_event,
            'financial_loss_inr': financial_loss,
            'customer_hist_orders_before_this': hist_orders,
            'customer_hist_rto_rate_before_this': round(hist_rto / hist_orders, 2) if hist_orders else None,
        })

        # update running history AFTER recording (so this order's features
        # reflect history BEFORE this order — avoids leakage)
        hist_orders += 1
        hist_rto += int(rto_triggered)
        hist_returns += int(post_delivery_return)
        hist_cod += int(payment_mode == 'cod')

    return orders


# =============================================================================
# SECTION 4: MAIN SIMULATION RUNNER
# =============================================================================

def run_simulation(n_customers=50_000):
    print("=" * 60)
    print("BUYER RELIABILITY INTELLIGENCE FRAMEWORK — Shiprocket")
    print("Generating simulation data...")
    print("=" * 60)

    persona_keys = list(PERSONAS.keys())
    persona_weights = [PERSONAS[p]['weight'] for p in persona_keys]
    assigned = np.random.choice(persona_keys, size=n_customers, p=persona_weights)

    customers = []
    all_orders = []

    print(f"\n[Step 1/3] Generating {n_customers:,} customer profiles...")
    for cid in range(1, n_customers + 1):
        pkey = assigned[cid - 1]
        customers.append(generate_customer(cid, pkey, PERSONAS[pkey]))

    print(f"[Step 2/3] Generating order events per customer...")
    for cust in customers:
        pkey = cust['persona']
        all_orders.extend(generate_orders_for_customer(cust, pkey, PERSONAS[pkey]))

    print(f"[Step 3/3] Building DataFrames + post-hoc ghost flag...")
    customers_df = pd.DataFrame(customers)
    orders_df = pd.DataFrame(all_orders)

    # ---------- Operational ghost flag (post-hoc, customer-level) ----------
    # Mirrors how you'd actually flag a ghost in production: based on
    # observed pattern across >=2 orders, not a single event.
    cust_summary = orders_df.groupby('customer_id').agg(
        total_orders=('order_id', 'count'),
        cod_orders=('payment_mode', lambda x: (x == 'cod').sum()),
        rto_orders=('rto_triggered', 'sum'),
        ghost_events=('is_ghost_event', 'sum'),
        avg_pickup_rate=('picked_up_pre_delivery_call', 'mean'),
    ).reset_index()
    cust_summary['cod_share'] = cust_summary['cod_orders'] / cust_summary['total_orders']
    cust_summary['rto_rate'] = cust_summary['rto_orders'] / cust_summary['total_orders']
    cust_summary['is_flagged_ghost'] = (
        (cust_summary['total_orders'] >= 2) &
        (cust_summary['rto_rate'] > 0.5) &
        (cust_summary['avg_pickup_rate'] < 0.4) &
        (cust_summary['cod_share'] > 0.7)
    )

    customers_df = customers_df.merge(
        cust_summary[['customer_id', 'total_orders', 'rto_rate', 'is_flagged_ghost']],
        on='customer_id', how='left'
    )

    # ---------- Summary stats ----------
    print("\n" + "=" * 60)
    print("SIMULATION SUMMARY")
    print("=" * 60)
    print(f"Total Customers:        {len(customers_df):>10,}")
    print(f"Total Orders:           {len(orders_df):>10,}")
    print(f"Overall RTO Rate:       {orders_df['rto_triggered'].mean()*100:>9.2f}%")
    print(f"Overall NDR Rate:       {orders_df['ndr_triggered'].mean()*100:>9.2f}%")
    print(f"Post-Delivery Return Rate: {orders_df['post_delivery_return'].mean()*100:>6.2f}%")
    print(f"Flagged Ghost Customers: {customers_df['is_flagged_ghost'].sum():>9,} "
          f"({customers_df['is_flagged_ghost'].mean()*100:.2f}% of customers)")
    print(f"Total Financial Loss (RTO, Cr): {orders_df['financial_loss_inr'].sum()/1e7:>6.2f}")

    print("\nRTO Rate by Payment Mode:")
    print(orders_df.groupby('payment_mode')['rto_triggered'].mean().mul(100).round(2))

    print("\nRTO Rate by City Tier x Payment Mode:")
    print(orders_df.groupby(['city_tier', 'payment_mode'])['rto_triggered'].mean().mul(100).round(2))

    print("\nReturn Rate by Category:")
    print(orders_df.groupby('category')['post_delivery_return'].mean().mul(100).round(2))

    orders_df.to_csv('shiprocket_orders.csv', index=False)
    customers_df.to_csv('shiprocket_customers.csv', index=False)

    print("\n✅ Files saved:")
    print("   → shiprocket_orders.csv")
    print("   → shiprocket_customers.csv")
    print("=" * 60)

    return customers_df, orders_df


if __name__ == '__main__':
    customers_df, orders_df = run_simulation(n_customers=50_000)
