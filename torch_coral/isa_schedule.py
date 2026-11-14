"""Relocate scalar loops/branches when shortening verified SRAM load waits."""
from functools import lru_cache


@lru_cache(maxsize=256)
def compact_load_waits(words):
    from coral.isa import WORD,opcode
    from .isa_scalar import _parts
    _,_,_,S=_parts()
    words=tuple(words); kept=[]; positions={}; index=0
    while index<len(words):
        positions[index]=len(kept)
        word=words[index]; kept.append((index,word)); index+=1
        if opcode(word)==0x20 and WORD.decode([word])['enable_vector'] in (2,3):
            end=index
            while end<len(words) and words[end]==S.NOP: end+=1
            # One separating bundle covers a two-cycle scalar load. SFU waits
            # are never shortened. References to removed bundles map forward.
            if end-index>=4:
                positions[index]=len(kept); kept.append((index,S.NOP)); index+=1
                while index<end: positions[index]=len(kept); index+=1
    positions[len(words)]=len(kept)
    result=[]
    for old,word in kept:
        op=opcode(word)
        if op==0x23:
            count=(word>>12)&0xffff; length=word>>28
            word=S.loop(count,positions[old+1+length]-positions[old+1])
        elif op==0x22:
            offset=(word>>17)&0x7fff
            if offset&0x4000: offset-=0x8000
            word=S.branch(positions[old+offset]-positions[old])
        result.append(word)
    return tuple(result)
