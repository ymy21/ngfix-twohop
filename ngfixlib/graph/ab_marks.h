#pragma once
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace ngfixlib {
enum class ABMode { NONE, A, B };
inline ABMode parseABMode(const std::string& s) {
    if(s=="none") return ABMode::NONE;
    if(s=="A") return ABMode::A;
    if(s=="B") return ABMode::B;
    throw std::runtime_error("AB mode must be none, A or B");
}
inline void abRead(std::istream& in, void* p, size_t bytes) {
    if(bytes && !in.read(static_cast<char*>(p),bytes)) throw std::runtime_error("Truncated AB input");
}
inline void abWrite(std::ostream& out, const void* p, size_t bytes) {
    if(bytes && !out.write(static_cast<const char*>(p),bytes)) throw std::runtime_error("AB output write failed");
}
// Existing raw index format is native 64-bit little-endian. Validate its full
// record framing BEFORE calling the legacy constructor (which ignores EOF).
inline void validateABIndex(const std::string& path) {
    static_assert(sizeof(size_t)==8 && sizeof(unsigned)==4);
    const uint32_t endian=1;
    if(*reinterpret_cast<const uint8_t*>(&endian)!=1) throw std::runtime_error("Little-endian host required");
    std::ifstream in(path,std::ios::binary);
    std::array<uint64_t,7> h{}; abRead(in,h.data(),56);
    const auto bytes=std::filesystem::file_size(path);
    const auto n=h[3], dim=h[5], capacity=h[6];
    if(!n || n>capacity || n>=UINT32_MAX || h[4]>=n || !dim || dim>65536 ||
       capacity>(UINT64_MAX-56)/(dim*4+1)) throw std::runtime_error("Invalid raw graph header");
    uint64_t pos=56+capacity*(dim*4+1);
    if(pos>bytes || n>(bytes-pos)/4) throw std::runtime_error("Incomplete raw float index");
    in.seekg(pos);
    for(uint64_t u=0;u<n;++u) {
        uint8_t meta[4]; abRead(in,meta,4); pos+=4;
        const auto cap=meta[0], sz=meta[1], repair_cap=meta[2], repair_sz=meta[3];
        if(sz>cap || repair_cap>cap || repair_sz>repair_cap || repair_sz>sz ||
           unsigned(sz-repair_sz)>unsigned(cap-repair_cap) || repair_sz)
            throw std::runtime_error("AB requires valid unrepaired graph records");
        const uint64_t payload=uint64_t(cap)*4+uint64_t(repair_cap)*2;
        if(payload>bytes-pos) throw std::runtime_error("Truncated graph adjacency");
        pos+=payload; in.seekg(pos);
    }
    if(pos!=bytes) throw std::runtime_error("Unexpected trailing graph data/format");
}
struct ABMarks {
    // Flags: seen=1, A=2, B=4. B implies A. Slot order is bound to fingerprint.
    uint64_t n=0, dim=0, entry=0, fingerprint=0, top_l=0, queries=0;
    std::vector<uint64_t> offsets;
    std::vector<uint8_t> flags;
    static uint64_t hashBytes(uint64_t h,const void* data,size_t bytes) {
        const auto* p=static_cast<const uint8_t*>(data);
        for(size_t i=0;i<bytes;++i) { h^=p[i]; h*=UINT64_C(1099511628211); }
        return h;
    }
    template<class Graph> static ABMarks layout(Graph& graph) {
        ABMarks marks; marks.n=graph.n.load(); marks.dim=graph.dim; marks.entry=graph.entry_point;
        marks.offsets.reserve(marks.n+1); marks.offsets.push_back(0);
        uint64_t h=UINT64_C(14695981039346656037);
        const uint64_t head[]={marks.n,marks.dim,marks.entry,graph.max_elements};
        h=hashBytes(h,head,sizeof(head));
        // Include vector bytes and deletion flags, not just topology.
        h=hashBytes(h,graph.vecdata,graph.max_elements*graph.size_per_element);
        for(size_t u=0;u<marks.n;++u) {
            if(graph.is_deleted(u)) throw std::runtime_error("AB stage 1 requires no deleted nodes");
            if(GET_NGFIX_SZ(reinterpret_cast<uint8_t*>(graph.Graph[u].neighbors)))
                throw std::runtime_error("AB stage 1 requires an unrepaired graph");
            auto [ids,degree,start]=graph.getNeighbors(u);
            const uint64_t count=degree; h=hashBytes(h,&count,sizeof(count));
            for(int j=start;j<start+degree;++j) {
                if(ids[j]>=marks.n) throw std::runtime_error("Graph neighbor out of range");
                h=hashBytes(h,&ids[j],sizeof(ids[j]));
            }
            marks.offsets.push_back(marks.offsets.back()+degree);
        }
        marks.fingerprint=h;
        return marks;
    }
    uint8_t at(uint32_t cur,size_t ordinal) const { return flags[offsets[cur]+ordinal]; }
    static bool allows(uint8_t flag,ABMode mode) {
        return mode==ABMode::NONE || (flag & (mode==ABMode::A?2:4));
    }
    void save(const std::string& path) const {
        if(std::filesystem::exists(path)) throw std::runtime_error("Refusing to overwrite marks: "+path);
        std::ofstream out(path,std::ios::binary);
        const char magic[8]={'A','B','M','A','R','K','0','1'};
        const uint64_t header[]={n,dim,entry,uint64_t(flags.size()),top_l,queries,fingerprint};
        abWrite(out,magic,8); abWrite(out,header,sizeof(header));
        abWrite(out,offsets.data(),offsets.size()*8); abWrite(out,flags.data(),flags.size());
        out.flush(); if(!out) throw std::runtime_error("Marks flush failed");
    }
    static ABMarks load(const std::string& path,const ABMarks& expected) {
        std::ifstream in(path,std::ios::binary); char magic[8]; uint64_t h[7];
        abRead(in,magic,8); abRead(in,h,sizeof(h));
        if(std::memcmp(magic,"ABMARK01",8) || h[0]!=expected.n || h[1]!=expected.dim ||
           h[2]!=expected.entry || h[3]!=expected.offsets.back() || h[6]!=expected.fingerprint || !h[4] || !h[5])
            throw std::runtime_error("AB marks/index identity mismatch");
        ABMarks m=expected; m.top_l=h[4]; m.queries=h[5];
        if(std::filesystem::file_size(path)!=64+(m.n+1)*8+h[3]) throw std::runtime_error("Invalid marks file length");
        std::vector<uint64_t> offsets(m.n+1); abRead(in,offsets.data(),offsets.size()*8);
        if(offsets!=m.offsets) throw std::runtime_error("AB slot offsets mismatch");
        m.flags.resize(h[3]); abRead(in,m.flags.data(),m.flags.size());
        for(auto f:m.flags) if(f!=0 && f!=1 && f!=3 && f!=7) throw std::runtime_error("Invalid A/B flag");
        return m;
    }
};

// Label a single exact top-L list. This helper is also tested on toy graphs.
// It writes per-query existential hits, not one increment per endpoint.
template<class Graph> void labelABQuery(Graph& graph,const std::vector<uint32_t>& top,
        const ABMarks& layout,std::vector<int32_t>& rank,std::vector<uint32_t>& excluded,
        uint32_t& stamp,std::vector<uint32_t>& opportunities,
        std::vector<uint32_t>& a_hits,std::vector<uint32_t>& b_hits) {
    const size_t L=top.size();
    if(!L || L>INT32_MAX) throw std::runtime_error("Invalid top-L size");
    for(size_t r=0;r<L;++r) {
        if(top[r]>=layout.n || rank[top[r]]!=-1) throw std::runtime_error("Invalid/duplicate GT node");
        rank[top[r]]=int32_t(r);
    }
    std::vector<std::vector<uint32_t>> local(L);
    for(size_t r=0;r<L;++r) {
        auto [ids,degree,start]=graph.getNeighbors(top[r]);
        for(int j=start;j<start+degree;++j) if(rank[ids[j]]>=0) local[r].push_back(rank[ids[j]]);
    }
    std::vector<uint8_t> reachable(L); std::vector<uint32_t> queue; queue.reserve(L);
    for(size_t r=0;r<L;++r) {
        if(++stamp==0) { std::fill(excluded.begin(),excluded.end(),0); stamp=1; }
        const uint32_t cur=top[r]; auto [mids,degree,start]=graph.getNeighbors(cur);
        excluded[cur]=stamp;
        for(int j=start;j<start+degree;++j) excluded[mids[j]]=stamp;
        bool reached=false;
        for(int j=start;j<start+degree;++j) {
            const uint32_t mid=mids[j]; if(rank[mid]>=0) continue;
            const size_t eid=layout.offsets[cur]+(j-start); ++opportunities[eid];
            auto [nns,ndeg,nstart]=graph.getNeighbors(mid);
            bool a=false,b=false;
            for(int k=nstart;k<nstart+ndeg;++k) {
                const uint32_t nn=nns[k]; const int32_t nr=rank[nn];
                if(nr<0 || excluded[nn]==stamp) continue;
                a=true;
                if(!reached) {
                    std::fill(reachable.begin(),reachable.end(),0); queue.clear();
                    reachable[r]=1; queue.push_back(uint32_t(r));
                    for(size_t p=0;p<queue.size();++p) for(auto v:local[queue[p]])
                        if(!reachable[v]) { reachable[v]=1; queue.push_back(v); }
                    reached=true;
                }
                if(!reachable[nr]) { b=true; break; }
            }
            if(a) ++a_hits[eid]; if(b) ++b_hits[eid];
        }
    }
    for(auto id:top) rank[id]=-1;
}
} // namespace ngfixlib
