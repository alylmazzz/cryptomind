export interface Signal {
  symbol: string
  exchange: string
  price: number
  direction: 'LONG' | 'SHORT' | 'FLAT'
  bias: string
  confidence: number
  actionable: boolean
  entry: number
  stop_loss: number
  take_profits: number[]
  risk_reward: number
  buy_pressure_pct: number
  sell_pressure_pct: number
  buy_threshold_min?: number
  buy_threshold_max?: number
  sell_threshold_min?: number
  sell_threshold_max?: number
  momentum_score?: number
  estimated_duration?: string
  position_size_pct?: number
  forecast?: {
    expected_high?: number
    expected_low?: number
    prob_up?: number
    up_move_pct?: number
    down_move_pct?: number
    band95?: [number, number]
  }
  reasons: string[]
  layer_breakdown: LayerBreakdown[]
  timeframe: string
  volatility?: 'low' | 'medium' | 'high' | 'extreme'
  correlation?: { symbol: string; value: number }
  sparkline?: number[]
}

export interface LayerBreakdown {
  layer: string
  score: number
  weight: number
  contribution: number
  detail?: Record<string, unknown>
}

export interface MarketClock {
  exchange: string
  label: string
  time: string
  status: 'open' | 'closed' | 'pre-market' | 'after-hours'
  flag?: string
}

export interface NewsItem {
  id: string
  title: string
  source: string
  sentiment: 'panic' | 'negative' | 'neutral' | 'positive' | 'euphoric'
  score: number
  time: string
  url?: string
  affected_symbols?: string[]
}

export interface MacroEvent {
  id: string
  category: string
  event: string
  impact: 'critical' | 'high' | 'medium'
  time: string
  t_minus?: string
  scenarios?: { type: 'bull' | 'base' | 'bear'; probability: number; description: string }[]
  correlation_chain?: string
  historical_pattern?: string
}

export interface CorrelationPair {
  symbol_a: string
  symbol_b: string
  value: number
  strength: 'strong_positive' | 'weak_positive' | 'neutral' | 'weak_negative' | 'strong_negative'
}
