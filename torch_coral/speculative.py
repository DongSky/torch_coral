"""Greedy 2..4-position verification; drafts never bypass the full TPU trunk."""
import time


def ngram_tail(history, prefix, count, minimum_width=2, excluded=()):
    """Propose known token IDs by suffix lookup; every result is later verified."""
    for width in range(min(6,len(prefix)),minimum_width-1,-1):
        suffix=prefix[-width:]
        for start in range(len(history)-width-count,-1,-1):
            if history[start:start+width]==suffix:
                tail=history[start+width:start+width+count]
                if not any(token in excluded for token in tail):
                    return tail
    return None


class SpeculativeGreedy:
    def __init__(self,engine,processed,*,adaptive=True,max_block_tokens=None):
        self.engine=engine
        self.processed=list(processed)
        self.adaptive=adaptive
        self.enabled=True
        maximum=getattr(engine.backend,'speculative_max_block_tokens',2) if max_block_tokens is None else max_block_tokens
        if maximum not in (2,3,4): raise ValueError('Verification blocks must contain 2, 3 or 4 tokens')
        if maximum>2 and not all(hasattr(engine,name) for name in ('verify_block','finish_block','mtp_draft_chain')):
            raise ValueError('Engine does not support multi-position verification')
        self.maximum=maximum
        self._next_block=2 if adaptive else maximum
        self._ngram_cooldown=0
        self._requested_size=2
        self.stats=dict(proposals=0,verified_pairs=0,accepted=0,rejected=0,
                        draft_seconds=0.,verify_seconds=0.,fallback_steps=0,disabled=False,
                        baseline_seconds=None,speculative_seconds=0.,speculative_tokens=0,speculative_blocks=0,
                        shortlist_rows=0,maximum_block_tokens=maximum,next_block_tokens=self._next_block,
                        block_profiles={},discarded_drafts=0,ngram_drafts=0)
        self.stats['baseline_source']='unmeasured'
        engine.speculative_stats=self.stats

    def advance(self,anchor,remaining,stop_ids):
        if remaining<1: return [],[],False
        if self.adaptive and self.stats['baseline_seconds'] is None and getattr(self.engine.backend,'reuse_decode_calibration',False):
            record=self.engine.backend.decode_calibration.get(self.engine.position//64)
            if record is not None and time.perf_counter()-record[1]<60:
                self.stats['baseline_seconds']=record[0];self.stats['baseline_source']='recent same-device measurement'
        if self.adaptive and self.stats['baseline_seconds'] is None:
            position=self.engine.position
            started=time.perf_counter(); scores=self.engine.step(anchor); selected=self.engine.greedy(scores)
            self.stats['baseline_seconds']=time.perf_counter()-started
            self.stats['baseline_source']='fresh ordinary step'
            self._remember_ordinary(position,self.stats['baseline_seconds'])
            self.stats['fallback_steps']+=1; self.processed.append(anchor)
            return ([] if selected in stop_ids else [selected]),[scores],selected in stop_ids
        testing=self.enabled and remaining>=2 and self.engine.position+2<=self.engine.context
        accepted_before=self.stats['accepted']
        started=time.perf_counter()
        result=self._advance(anchor,remaining,stop_ids)
        if testing:
            self.stats['speculative_seconds']+=time.perf_counter()-started
            self.stats['speculative_tokens']+=len(result[0]); self.stats['speculative_blocks']+=1
            key=str(self._requested_size)
            profiles=self.stats['block_profiles']
            profile=profiles.setdefault(key,dict(blocks=0,seconds=0.,visible_tokens=0))
            profile['blocks']+=1; profile['seconds']+=time.perf_counter()-started; profile['visible_tokens']+=len(result[0])
            accepted=self.stats['accepted']-accepted_before
            if self.adaptive and self.maximum>2:
                pair=profiles.get('2')
                pair_cost=(pair['seconds']/max(1,pair['visible_tokens'])) if pair else self.stats['baseline_seconds']
                if self._requested_size>2 and profile['seconds']/max(1,profile['visible_tokens'])>=pair_cost:
                    self._ngram_cooldown=12
                elif self._ngram_cooldown:
                    self._ngram_cooldown-=1
                self._next_block=2 if self._ngram_cooldown else self.maximum
            elif (self.adaptive and self._requested_size<self.maximum and accepted==self._requested_size-1
                    and len(result[0])==self._requested_size and str(self._requested_size+1) not in profiles):
                self._next_block=self._requested_size+1
            elif self.adaptive:
                self._next_block=int(min(profiles,key=lambda size:profiles[size]['seconds']/max(1,profiles[size]['visible_tokens'])))
            self.stats['next_block_tokens']=self._next_block
            rate=self.stats['speculative_seconds']/max(1,self.stats['speculative_tokens'])
            if (self.adaptive and self.stats['speculative_blocks']>=2 and
                    rate>=self.stats['baseline_seconds']):
                self.enabled=False; self.stats['disabled']=True
        return result

    def _remember_ordinary(self,position,seconds):
        if getattr(self.engine.backend,'reuse_decode_calibration',False):
            self.engine.backend.decode_calibration[position//64]=(seconds,time.perf_counter())

    def _advance(self,anchor,remaining,stop_ids):
        """Return verified output IDs, corresponding full logits, and stop status."""
        e=self.engine
        self._requested_size=min(self._next_block if self.adaptive else self.maximum,remaining,e.context-e.position)
        if len(self.processed)!=e.position: raise ValueError('Speculative token/cache alignment mismatch')
        if not self.enabled or remaining<2 or e.position+2>e.context:
            started=time.perf_counter();position=e.position
            scores=e.step(anchor); self.processed.append(anchor); self.stats['fallback_steps']+=1
            selected=e.greedy(scores)
            self._remember_ordinary(position,time.perf_counter()-started)
            return ([] if selected in stop_ids else [selected]),[scores],selected in stop_ids
        if self._requested_size>2:
            return self._advance_block(anchor,self._requested_size,stop_ids)
        started=time.perf_counter()
        pool=getattr(e.backend,'draft_candidate_ids',None)
        if pool and hasattr(e.backend,'project_indexed'):
            ids=list(dict.fromkeys([*pool,*self.processed[-32:],anchor,*sorted(stop_ids)]))
            draft=ids[e.greedy(e.mtp_draft(self.processed+[anchor],candidate_ids=ids))]
            self.stats['shortlist_rows']+=len(ids)
        else: draft=e.greedy(e.mtp_draft(self.processed+[anchor]))
        self.stats['draft_seconds']+=time.perf_counter()-started
        self.stats['proposals']+=1
        if draft in stop_ids:
            # There is no useful second visible candidate to speculate on.
            scores=e.step(anchor); self.processed.append(anchor); self.stats['fallback_steps']+=1
            selected=e.greedy(scores)
            return ([] if selected in stop_ids else [selected]),[scores],selected in stop_ids
        started=time.perf_counter(); scores=e.verify_pair(anchor,draft)
        wanted=e.greedy(scores[0]); accepted=draft==wanted
        e.finish_pair(accepted)
        self.stats['verify_seconds']+=time.perf_counter()-started
        self.stats['verified_pairs']+=1
        if accepted:
            self.stats['accepted']+=1
            self.processed.extend([anchor,draft])
            bonus=e.greedy(scores[1])
            return ([draft] if bonus in stop_ids else [draft,bonus]),[scores[0],scores[1]],bonus in stop_ids
        self.stats['rejected']+=1
        self.processed.append(anchor)
        return ([] if wanted in stop_ids else [wanted]),[scores[0]],wanted in stop_ids

    def _advance_block(self,anchor,size,stop_ids):
        e=self.engine
        pool=getattr(e.backend,'draft_candidate_ids',None)
        ids=list(dict.fromkeys([*pool,*self.processed[-32:],anchor,*sorted(stop_ids)])) if pool and hasattr(e.backend,'project_indexed') else None
        begin=time.perf_counter()
        drafts=e.mtp_draft_chain(self.processed+[anchor],size-1,candidate_ids=ids,stop_ids=stop_ids,
                                 allow_estimates=not self.adaptive)
        self.stats['draft_seconds']+=time.perf_counter()-begin
        self.stats['proposals']+=len(drafts)
        self.stats['shortlist_rows']+=len(ids or [])*(len(drafts)-getattr(e,'last_ngram_drafts',0))
        self.stats['ngram_drafts']+=getattr(e,'last_ngram_drafts',0)
        if any(token in stop_ids for token in drafts):
            drafts=drafts[:next(i for i,token in enumerate(drafts) if token in stop_ids)]
        self._requested_size=max(2,len(drafts)+1)
        if not drafts:
            scores=e.step(anchor);self.processed.append(anchor);self.stats['fallback_steps']+=1
            selected=e.greedy(scores)
            return ([] if selected in stop_ids else [selected]),[scores],selected in stop_ids
        begin=time.perf_counter()
        try:
            scores=e.verify_block(anchor,drafts)
            emitted=[];logits=[];committed=1;accepted=0;stopped=False
            for index,draft in enumerate(drafts):
                selected=e.greedy(scores[index]);logits.append(scores[index])
                if selected in stop_ids:
                    self.stats['rejected']+=1
                    stopped=True;break
                emitted.append(selected)
                if selected!=draft:
                    self.stats['rejected']+=1
                    break
                accepted+=1;committed+=1
            else:
                selected=e.greedy(scores[-1]);logits.append(scores[-1])
                stopped=selected in stop_ids
                if not stopped:emitted.append(selected)
            e.finish_block(committed)
        except BaseException:
            if getattr(e,'verified_prefix_caches',None) is not None:
                e.finish_block(0)
            raise
        self.processed.extend([anchor,*drafts[:committed-1]])
        self.stats['verify_seconds']+=time.perf_counter()-begin
        self.stats['verified_pairs']+=len(drafts)
        self.stats['accepted']+=accepted
        self.stats['discarded_drafts']+=len(drafts)-accepted-int(accepted<len(drafts))
        return emitted,logits,stopped
